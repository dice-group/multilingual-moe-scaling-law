"""Shared contracts, paths, hashing, and provenance for A/B-v2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.repo import find_repo_root


REPO_ROOT = find_repo_root(Path(__file__)).resolve()
DEFAULT_SPEC = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/"
    "study_spec_jointmoe_paper_lr_v1.json"
)
DEFAULT_RUNTIME = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/"
    "workflow_runtime_jointmoe_paper_lr_v1.json"
)
DEFAULT_OUTPUT = Path("outputs/pipeline_runs/h100_ab_v2_jointmoe_paper_lr_v1")
MANIFEST_NAME = "launch_manifest.json"
STATE_NAME = "workflow_state.json"
GPU_ESTIMATE_NAME = "gpu_hour_estimate.json"
DATA_ARTIFACT_SEAL_NAME = "data_artifact_seal.json"
RECOVERY_AUTHORIZATION_ENV = "PROOF_AB_V2_RECOVERY_AUTHORIZATION"
RECOVERY_AUTHORIZATION_SHA256_ENV = (
    "PROOF_AB_V2_RECOVERY_AUTHORIZATION_SHA256"
)
WORKFLOW_SCHEMA_VERSION = 2
SUBMISSION_PHASES = (
    "pilot",
    "tokenizer",
    "corpus",
    "tokenization",
    "sentinel",
    "lr_verification",
    "lr_diagnostics",
    "production",
)
RETRYABLE_TERMINAL_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}
EXPERTS = (1, 4, 16, 32)
LANGUAGE_COUNTS = (1, 12, 50, 200)
LADDERS = ("A", "B")
VALIDATION_CANDIDATE_FRACTION = 0.30
VALIDATION_TARGET_FRACTION = 0.20
# Compatibility authority for the already-frozen C/D workflows. A/B no longer
# consumes this value; its rates come from JOINTMOE_PAPER_LR_* below.
FIXED_PEAK_LEARNING_RATE = 2e-4
JOINTMOE_PAPER_LR_INTERCEPT = 8.39
JOINTMOE_PAPER_LR_ACTIVE_COEFFICIENT = -0.81
JOINTMOE_PAPER_LR_EXPERT_COEFFICIENT = -0.25
PRIMARY_TRAINING_SEED = 1234
REPLICATION_TRAINING_SEEDS = (1234, 5678, 91011)
PRIMARY_METRIC = "equal_language_atlas_tao_adjusted_loss"
EXPECTED_MICRO_BATCH_SIZE = {
    "A": {"1": 64, "4": 64, "16": 64, "32": 64},
    "B": {"1": 64, "4": 64, "16": 64, "32": 32},
}
REQUIRED_SENTINEL_PROBES = {
    ("early_correctness", "A", 1, 32),
    ("early_correctness", "A", 32, 32),
    ("early_correctness", "B", 16, 32),
    ("early_correctness", "B", 32, 32),
    ("plateau_memory_perf", "A", 32, 128),
    ("plateau_memory_perf", "B", 16, 128),
    ("plateau_memory_perf", "B", 32, 128),
    ("mid_ramp_ga2", "B", 32, 64),
}
H100_PEAK_TFLOPS_BF16 = 989.5
SENTINEL_INTEGRATION_LANGUAGE = "eng_Latn"
LOSS_RE = re.compile(r"\blm loss:\s*([^\s|]+)", re.IGNORECASE)
GRAD_NORM_RE = re.compile(r"\bgrad norm:\s*([^\s|]+)", re.IGNORECASE)
LEARNING_RATE_RE = re.compile(r"\blearning rate:\s*([^\s|]+)", re.IGNORECASE)
SKIPPED_RE = re.compile(r"\bnumber of skipped iterations:\s*(\d+)", re.IGNORECASE)
NAN_ITER_RE = re.compile(r"\bnumber of nan iterations:\s*(\d+)", re.IGNORECASE)
OPTIONAL_DEPENDENCY_PREFIX = "[bridge_optional_dependencies] "
PRESCRIBED_LR_SCHEDULE_PROBE_TOKENS = 167_772_160
PRESCRIBED_LR_SCHEDULE_WARMUP_TOKENS = 130_023_424
PRESCRIBED_LR_SCHEDULE_WARMUP_STEPS = 1_984
PRESCRIBED_LR_SCHEDULE_PROBE_CELLS = (
    ("A", 1, 1),
    ("A", 1, 200),
    ("B", 32, 1),
    ("B", 32, 200),
)
# Historical C/D wrappers override this alias with their frozen probe cells.
FIXED_LR_SCHEDULE_PROBE_TOKENS = PRESCRIBED_LR_SCHEDULE_PROBE_TOKENS
FIXED_LR_SCHEDULE_WARMUP_TOKENS = PRESCRIBED_LR_SCHEDULE_WARMUP_TOKENS
FIXED_LR_SCHEDULE_WARMUP_STEPS = PRESCRIBED_LR_SCHEDULE_WARMUP_STEPS
FIXED_LR_SCHEDULE_PROBE_CELLS = PRESCRIBED_LR_SCHEDULE_PROBE_CELLS
REQUIRED_TRAINING_ENV_KEYS = {
    "BRIDGE_LOG_THROUGHPUT",
    "BRIDGE_THROUGHPUT_WINDOW_SIZE",
    "BRIDGE_MOE_TOKEN_DROP_METRICS",
    "BRIDGE_PREFER_VENV_CUDA_LIBS",
    "PROOF_ENABLE_BRIDGE_OPTIONAL_DEP_SHIMS",
    "PROOF_REQUIRE_REAL_MODELOPT",
    "PROOF_GPU_MONITOR_INTERVAL_SECONDS",
    "PROOF_DATASET_BUILDER_THREADS",
    "OMP_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN",
    "NVTE_UNFUSED_ATTN",
    "LD_LIBRARY_PATH",
    "TMPDIR",
    "TMP",
    "TEMP",
}
REQUIRED_DATA_ENV_KEYS = {
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "TOKENIZERS_PARALLELISM",
}


class WorkflowError(RuntimeError):
    """A fail-closed workflow contract violation."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorkflowError(f"required JSON file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkflowError(f"expected a JSON object: {path}")
    inherited = payload.pop("inherits", None)
    if inherited is None:
        return payload
    expected_parent_sha256: str | None = None
    if isinstance(inherited, Mapping):
        expected_parent_sha256 = inherited.get("spec_sha256")
        inherited = inherited.get("spec_path")
    if not isinstance(inherited, str) or not inherited.strip():
        raise WorkflowError(f"JSON inheritance path must be a non-empty string: {path}")
    parent_path = Path(inherited)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    parent_path = parent_path.resolve()
    if expected_parent_sha256 is not None and (
        not isinstance(expected_parent_sha256, str)
        or _file_sha256(parent_path) != expected_parent_sha256
    ):
        raise WorkflowError(f"JSON inheritance checksum mismatch: {parent_path}")
    parent = _load_object(parent_path)
    return _merge_inherited_object(parent, payload, source=path)


def _merge_inherited_object(
    base: Mapping[str, Any], overlay: Mapping[str, Any], *, source: Path
) -> dict[str, Any]:
    """Apply the small, explicit config-inheritance format used by later ladders.

    ``{"__replace__": value}`` replaces a mapping/list rather than recursively
    merging it.  This keeps a C-only axis from accidentally retaining A/B
    entries while preserving the shared corpus/runtime policy by default.
    """

    merged = {str(key): value for key, value in base.items()}
    for key, value in overlay.items():
        if isinstance(value, Mapping) and set(value) == {"__delete__"}:
            if value["__delete__"] is not True:
                raise WorkflowError(f"invalid __delete__ directive in {source}")
            merged.pop(str(key), None)
            continue
        if isinstance(value, Mapping) and set(value) == {"__replace__"}:
            merged[str(key)] = value["__replace__"]
            continue
        existing = merged.get(str(key))
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[str(key)] = _merge_inherited_object(existing, value, source=source)
            continue
        merged[str(key)] = value
    return merged


def _is_h100_resource(resource: object) -> bool:
    """Recognize one- and multi-GPU H100 resource classes."""

    return str(resource).startswith("h100")


def _workflow_profile_name(runtime: Mapping[str, Any]) -> str:
    """Return the immutable local pipeline profile name for this study ladder."""

    value = runtime.get("workflow_profile_name", "ab_v2_job")
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError("runtime workflow_profile_name must be a non-empty string")
    return value


def _workflow_attempt_id(runtime: Mapping[str, Any], *, role: str) -> str:
    value = runtime.get("workflow_attempt_id", "ab_v2")
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError("runtime workflow_attempt_id must be a non-empty string")
    return f"{value}_{role}"


def _runtime_primary_training_seed(runtime: Mapping[str, Any]) -> int:
    seeds = runtime.get("seeds", {})
    if not isinstance(seeds, Mapping):
        return PRIMARY_TRAINING_SEED
    return int(seeds.get("primary_training", seeds.get("model", PRIMARY_TRAINING_SEED)))


def _learning_rate_prescription(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the literal rounded-coefficient JointMoE paper law for A/B."""

    # Import lazily: ``policy`` initializes from this module's base constants.
    from .variant import active_policy

    policy = active_policy()

    training = spec.get("training_policy")
    raw = training.get("learning_rate_prescription") if isinstance(training, Mapping) else None
    if not isinstance(raw, Mapping):
        raise WorkflowError("study spec lacks training_policy.learning_rate_prescription")
    expected = {
        "intercept": JOINTMOE_PAPER_LR_INTERCEPT,
        "active_parameter_coefficient": JOINTMOE_PAPER_LR_ACTIVE_COEFFICIENT,
        "expert_coefficient": JOINTMOE_PAPER_LR_EXPERT_COEFFICIENT,
        "constant_across_K": True,
        "released_code_full_precision_coefficients_used": False,
        "per_cell_selection_allowed": False,
    }
    drift = {
        key: {"expected": value, "observed": raw.get(key)}
        for key, value in expected.items()
        if raw.get(key) != value
    }
    if drift:
        raise WorkflowError(f"JointMoE paper LR prescription drifted: {drift}")
    ladders = spec.get("axes", {}).get("ladders")
    if not isinstance(ladders, list):
        raise WorkflowError("study spec lacks ladder parameter counts")
    counts: dict[str, int] = {}
    for row in ladders:
        if not isinstance(row, Mapping):
            raise WorkflowError("study spec ladder is not an object")
        ladder = str(row.get("id", ""))
        try:
            count = int(row["N_act_nonembed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(f"{ladder or 'unknown'} lacks N_act_nonembed") from exc
        if ladder not in policy.ladders:
            continue
        if count <= 0 or ladder in counts:
            raise WorkflowError(f"invalid JointMoE parameter-count row: {row}")
        counts[ladder] = count
    if set(counts) != set(policy.ladders):
        raise WorkflowError(
            f"JointMoE LR prescription lacks parameter counts for {policy.ladders}"
        )
    rates = {
        ladder: {
            str(experts): math.exp(
                JOINTMOE_PAPER_LR_INTERCEPT
                + JOINTMOE_PAPER_LR_ACTIVE_COEFFICIENT * math.log(count)
                + JOINTMOE_PAPER_LR_EXPERT_COEFFICIENT * math.log(experts)
            )
            for experts in EXPERTS
        }
        for ladder, count in counts.items()
    }
    return {
        **dict(raw),
        "N_act_nonembed_by_ladder": counts,
        "base_peak_learning_rate_by_ladder_and_experts": rates,
    }


def _prescribed_peak_learning_rate(
    prescription: Mapping[str, Any],
    *,
    ladder: str,
    experts: int,
    multiplier: float = 1.0,
) -> float:
    try:
        base = float(
            prescription["base_peak_learning_rate_by_ladder_and_experts"][ladder][
                str(experts)
            ]
        )
        multiplier = float(multiplier)
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(f"missing prescribed LR for {ladder}/E{experts}") from exc
    if not math.isfinite(base) or base <= 0 or not math.isfinite(multiplier) or multiplier <= 0:
        raise WorkflowError(f"invalid prescribed LR or multiplier for {ladder}/E{experts}")
    return base * multiplier


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _abs(path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _language_map(k: int) -> str:
    return f"configs/language_groupings/scaling_fw2_v2_k{k:03d}.json"


@dataclass(frozen=True, slots=True)
class Paths:
    """Workflow-owned outputs plus an optional immutable corpus source.

    A production relaunch must never need to reconstruct an already audited
    corpus merely because its training/evaluation implementation changed.  In
    that case ``root`` remains the new workflow root (jobs, receipts, logs,
    Triton caches), while every corpus read resolves below ``data_root``.
    """

    root: Path

    data_root: Path | None = None

    @property
    def corpus_root(self) -> Path:
        return self.data_root if self.data_root is not None else self.root

    @property
    def source_lock(self) -> Path:
        return self.corpus_root / "corpus" / "source_lock"

    @property
    def bootstrap(self) -> Path:
        return self.corpus_root / "corpus" / "bootstrap"

    @property
    def sentence_root(self) -> Path:
        return self.corpus_root / "corpus" / "tokenizer_sentences"

    @property
    def tokenizer_acquisition_plan(self) -> Path:
        return self.corpus_root / "corpus" / "tokenizer_sentence_acquisition_plan.json"

    @property
    def tokenizer_bootstrap(self) -> Path:
        return self.corpus_root / "corpus" / "tokenizer_bootstrap"

    @property
    def tokenizer(self) -> Path:
        return self.corpus_root / "corpus" / "tokenizer" / "sp_unigram_64k"

    @property
    def calibration(self) -> Path:
        return self.corpus_root / "corpus" / "token_calibration.json"

    @property
    def sampling_plan(self) -> Path:
        return self.corpus_root / "corpus" / "sampling_plan_32b.json"

    @property
    def production_bootstrap(self) -> Path:
        return self.corpus_root / "corpus" / "production_bootstrap"

    @property
    def release(self) -> Path:
        return self.corpus_root / "corpus" / "release_32b"

    @property
    def tokenized(self) -> Path:
        return self.corpus_root / "corpus" / "tokenized"

    @property
    def tokenization_work(self) -> Path:
        return self.corpus_root / "corpus" / "tokenization_work"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def control(self) -> Path:
        return self.root / "control"


def _paths_for_manifest(manifest: Mapping[str, Any]) -> Paths:
    """Resolve the immutable corpus source without moving workflow outputs."""

    output_root = Path(str(manifest["output_root"])).resolve()
    materialization = manifest.get("data_materialization")
    if not isinstance(materialization, Mapping):
        # Kept for narrow unit-level runner tests that construct only the
        # fields needed to exercise receipt ordering.
        return Paths(output_root)
    if materialization.get("mode") != "reuse_sealed_k12":
        return Paths(output_root)
    sealed = materialization.get("sealed_corpus")
    if not isinstance(sealed, Mapping) or not isinstance(sealed.get("root"), str):
        raise WorkflowError("sealed-corpus reuse manifest lacks its source root")
    source_root = Path(str(sealed["root"])).resolve()
    if source_root == output_root:
        raise WorkflowError("sealed corpus source must differ from the workflow root")
    return Paths(output_root, data_root=source_root)


from .common_provenance import (
    _config_inheritance_chain,
    _contract_files,
    _fixed_hashes,
    _git_diff_identity,
    _git_provenance,
    _git_snapshot,
    _tracked_git_provenance_identity,
    _workflow_source_files,
)


def _recovery_authorization(
    manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load a narrowly scoped, hash-bound workflow recovery grant.

    A recovery grant does not mutate or weaken the sealed launch manifest. It
    binds one exact replacement execution checkout, the complete Git diff from
    the launch commit, the fixed inputs that changed, and the only job IDs that
    may execute. Every other fixed input and the Bridge commit remain frozen.
    """

    raw_path = os.getenv(RECOVERY_AUTHORIZATION_ENV, "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).resolve()
    expected_file_sha = os.getenv(RECOVERY_AUTHORIZATION_SHA256_ENV, "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha):
        raise WorkflowError("recovery authorization SHA-256 is missing or malformed")
    actual_file_sha = _file_sha256(path)
    if actual_file_sha != expected_file_sha:
        raise WorkflowError(
            "recovery authorization file hash changed: "
            f"expected {expected_file_sha}, got {actual_file_sha}"
        )
    authorization = _load_object(path)
    expected_payload_sha = str(authorization.get("authorization_sha256", ""))
    candidate = dict(authorization)
    candidate.pop("authorization_sha256", None)
    if expected_payload_sha != _payload_sha256(candidate):
        raise WorkflowError("recovery authorization payload hash mismatch")
    if authorization.get("schema_version") != 1 or authorization.get("status") != "authorized":
        raise WorkflowError("recovery authorization is not an active schema-v1 grant")
    kind = authorization.get("kind")
    if kind not in {
        "legacy_c_v1_k200_provenance_recovery",
        "fixed_input_recovery_v1",
    }:
        raise WorkflowError("unsupported recovery authorization kind")
    if authorization.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise WorkflowError("recovery authorization targets a different manifest")
    if authorization.get("study_id") != manifest.get("study_id"):
        raise WorkflowError("recovery authorization targets a different study")
    if authorization.get("launch_git_provenance") != manifest.get("git_provenance"):
        raise WorkflowError("recovery authorization launch provenance changed")

    allowed_jobs = authorization.get("allowed_job_ids")
    if (
        not isinstance(allowed_jobs, list)
        or not allowed_jobs
        or any(not isinstance(value, str) or not value for value in allowed_jobs)
        or len(set(allowed_jobs)) != len(allowed_jobs)
    ):
        raise WorkflowError("recovery authorization job allowlist is malformed")
    manifest_jobs = {str(job["job_id"]) for job in manifest.get("jobs", [])}
    if not set(allowed_jobs) <= manifest_jobs:
        raise WorkflowError("recovery authorization contains unknown job IDs")

    execution_provenance = authorization.get("execution_git_provenance")
    if not isinstance(execution_provenance, Mapping):
        raise WorkflowError("recovery authorization lacks execution Git provenance")
    _tracked_git_provenance_identity(execution_provenance)

    diff_identity = authorization.get("execution_diff")
    if not isinstance(diff_identity, Mapping):
        raise WorkflowError("recovery authorization lacks execution diff identity")
    if kind == "legacy_c_v1_k200_provenance_recovery":
        if _tracked_git_provenance_identity(
            execution_provenance
        ) != _tracked_git_provenance_identity(_git_provenance()):
            raise WorkflowError(
                "legacy recovery execution repository/submodule provenance changed"
            )
        observed_diff = _git_diff_identity(
            REPO_ROOT,
            str(diff_identity.get("base_commit", "")),
            str(diff_identity.get("head_commit", "")),
        )
        if dict(diff_identity) != observed_diff:
            raise WorkflowError("legacy recovery execution Git diff identity changed")

    allowed_changes = authorization.get("allowed_fixed_input_changes")
    if not isinstance(allowed_changes, Mapping) or not allowed_changes:
        raise WorkflowError("recovery authorization fixed-input allowlist is malformed")
    return authorization


def _validate_recovery_job_authorization(
    manifest: Mapping[str, Any], job_id: str
) -> dict[str, Any] | None:
    authorization = _recovery_authorization(manifest)
    if authorization is None:
        return None
    if job_id not in authorization["allowed_job_ids"]:
        raise WorkflowError(f"recovery authorization does not permit job {job_id}")
    return authorization


def _recovery_receipt_identity(
    manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    authorization = _recovery_authorization(manifest)
    if authorization is None:
        return None
    return {
        "kind": authorization["kind"],
        "authorization_path": str(
            Path(os.environ[RECOVERY_AUTHORIZATION_ENV]).resolve()
        ),
        "authorization_file_sha256": os.environ[
            RECOVERY_AUTHORIZATION_SHA256_ENV
        ],
        "authorization_payload_sha256": authorization["authorization_sha256"],
        "launch_commit": authorization["execution_diff"]["base_commit"],
        "execution_commit": authorization["execution_diff"]["head_commit"],
    }


def _resource_gpu_count(runtime: Mapping[str, Any], resource_name: str) -> int:
    """Parse the requested GPU count from a frozen Slurm resource class."""

    # Focused projection tests intentionally pass only the measurement fields;
    # the real manifest builder validates the full resource object before this
    # helper is reached.  Preserve the historical one-H100 default there,
    # while requiring an explicit GRES for every real or multi-GPU workflow.
    raw_resources = runtime.get("resources")
    if not isinstance(raw_resources, Mapping) and resource_name == "h100":
        return 1
    try:
        gres = str(raw_resources[resource_name]["gres"])
        prefix, family, count = gres.split(":")
        resolved = int(count)
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(
            f"resource {resource_name!r} must use gres gpu:<family>:<positive-count>"
        ) from exc
    if prefix != "gpu" or not family or resolved <= 0:
        raise WorkflowError(
            f"resource {resource_name!r} has invalid GPU GRES {gres!r}"
        )
    return resolved


def _submission_phase(job: Mapping[str, Any]) -> str:
    job_id = str(job["job_id"])
    if job_id in {
        "storage_preflight",
        "data_source_lock",
        "data_metadata_preflight",
        "data_bootstrap_workers",
        "data_bootstrap_merge",
        "data_tokenizer_acquisition_plan",
    }:
        return "pilot"
    if job_id == "data_reuse_gate" or (
        job_id == "data_megatron_reader_smoke"
        and any(
            str(dependency) == "data_reuse_gate"
            for dependency in job.get("dependencies", [])
        )
    ):
        return "pilot"
    if job_id in {
        "data_tokenizer_metadata_preflight",
        "data_tokenizer_bootstrap_workers",
        "data_tokenizer_bootstrap_merge",
        "data_tokenizer_sentences",
        "data_tokenizer_train",
        "data_token_calibration",
        "data_sampling_plan",
    }:
        return "tokenizer"
    if job_id in {
        "data_production_metadata_preflight",
        "data_production_workers",
        "data_production_merge",
        "data_finalize",
    }:
        return "corpus"
    if job_id in {
        "data_tokenize_plan",
        "data_tokenize_workers",
        "data_tokenize_merge",
        "data_audit_gate",
        "data_megatron_reader_smoke",
    }:
        return "tokenization"
    if job.get("stage") == "sentinel" or job_id == "retention_prune_sentinels":
        return "sentinel"
    if job.get("stage") == "lr_verification":
        return "lr_verification"
    if job.get("stage") == "lr_diagnostics":
        return "lr_diagnostics"
    if job.get("stage") in {"production", "validation", "summary"} or job_id.startswith(
        "retention_prune_"
    ):
        return "production"
    # Unit-test/future synthetic jobs are allowed only through the direct API;
    # every real frozen workflow job must match above (validated in build).
    return "production"


def _submit_slot_count(job: Mapping[str, Any]) -> int:
    array = job.get("array")
    return len(_array_indices(str(array))) if array else 1


def _array_indices(spec: str) -> list[int]:
    """Expand the simple deterministic array syntax used by this workflow."""

    core = str(spec).split("%", 1)[0]
    result: list[int] = []
    for part in core.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            result.append(int(part))
            continue
        start_text, end_text = part.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if start < 0 or end < start:
            raise WorkflowError(f"invalid Slurm array range: {spec}")
        result.extend(range(start, end + 1))
    if not result or len(result) != len(set(result)):
        raise WorkflowError(f"invalid or duplicate Slurm array indices: {spec}")
    return result


def _job(manifest: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    matches = [row for row in manifest["jobs"] if row["job_id"] == job_id]
    if len(matches) != 1:
        raise WorkflowError(f"workflow job ID does not resolve exactly once: {job_id}")
    return dict(matches[0])


def _run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    completed = subprocess.run(
        list(cmd), cwd=REPO_ROOT, env=dict(env or os.environ), check=False
    )
    if completed.returncode:
        raise WorkflowError(
            f"command failed with exit code {completed.returncode}: {' '.join(cmd)}"
        )
