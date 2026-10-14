"""Validate Stage-1 benchmark campaigns and their checkpoint lineage."""

from __future__ import annotations

import copy
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.experiments.scaling.stage1_benchmarking.api import (
    _load_json,
    _resolve,
    _sha256_file,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_models import (
    resolve_checkpoint_models,
)
from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import (
    NLL_SUM_RELATIVE_TOLERANCE,
    RAW_NLL_ABS_TOLERANCE,
)

PROFILE_SCHEMA = "stage1_jointmoe_benchmark_profile/v3"
PROFILE_OVERLAY_SCHEMA = "stage1_jointmoe_benchmark_profile/v4"

BENCHMARK_ORDER = (
    "belebele_cloze",
    "okapi_hellaswag",
    "xstorycloze",
    "xcopa",
    "uhura_arc_easy",
    "global_mmlu_cloze",
)


@dataclass(frozen=True, slots=True)
class CampaignProfile:
    source_path: Path
    source_sha256: str
    payload: Mapping[str, Any]
    repo_root: Path
    harness_root: Path
    bridge_root: Path
    output_root: Path
    hf_home: Path
    lm_eval_python: Path
    bridge_python: Path

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(str(item["id"]) for item in self.payload["models"])

    @property
    def calibration_model_ids(self) -> tuple[str, ...]:
        return tuple(
            str(item["id"])
            for item in self.payload["models"]
            if item["timing_calibration"]
        )

    def model(self, model_id: str) -> Mapping[str, Any]:
        matches = [item for item in self.payload["models"] if item["id"] == model_id]
        if len(matches) != 1:
            raise ValueError(f"unknown or duplicate model ID: {model_id}")
        return matches[0]

    def request_set_name(self, model_id: str) -> str:
        request_set = self.model(model_id).get("request_set")
        if not isinstance(request_set, str) or request_set not in self.payload["request_sets"]:
            raise ValueError(f"invalid request set for {model_id}: {request_set!r}")
        return request_set

    def request_manifest(self, request_set: str) -> Path:
        if request_set not in self.payload["request_sets"]:
            raise ValueError(f"unknown request set: {request_set}")
        return self.output_root / "requests" / request_set / "frozen_requests.json"

    def score_spec(self, model_id: str) -> Path:
        request_set = self.request_set_name(model_id)
        return self.output_root / "specs" / request_set / f"{model_id}.score_spec.json"

    def scores(self, model_id: str) -> Path:
        return self.output_root / "scores" / model_id / "scores.json"

    def metrics_dir(self, model_id: str) -> Path:
        return self.output_root / "metrics" / model_id

    def runtime_receipt(self, model_id: str) -> Path:
        self.model(model_id)
        return self.output_root / "runtime" / f"{model_id}.json"

    def topology(self, model: Mapping[str, Any]) -> dict[str, Any]:
        experts = int(model["experts"])
        world_size = int(model["expert_parallel_size"])
        value = self.payload["evaluation_topology_by_world_size"].get(str(world_size))
        if not isinstance(value, dict):
            raise ValueError(f"missing evaluation topology for world size {world_size}")
        result = dict(value)
        result["micro_batch_size"] = int(model["micro_batch_size"])
        policy = self.payload["a40_model_override_policy"]
        overrides = dict(policy["base"])
        if experts > 1:
            overrides.update(policy["moe"])
        if world_size > 1:
            overrides.update(policy["expert_parallel"])
        result["a40_model_overrides"] = overrides
        return result

    def training_paths(self, model: Mapping[str, Any]) -> tuple[Path, Path, str]:
        base = self.source_path.parent
        training_manifest = _resolve(
            base, model.get("training_manifest"), field=f"{model.get('id')}.training_manifest"
        )
        evaluation_spec = _resolve(
            base, model.get("evaluation_spec"), field=f"{model.get('id')}.evaluation_spec"
        )
        run_id = model.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"invalid run_id for {model.get('id')}")
        return training_manifest, evaluation_spec, run_id


def _positive_int_list(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = tuple(_positive_int(item, field=field) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_sha(path: Path, expected: Any, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} SHA-256 is invalid")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash drift: expected {expected}, got {actual}")


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_profile_payload(source_path: Path) -> tuple[dict[str, Any], dict[str, str] | None]:
    raw_payload = _load_json(source_path, label="JointMoE benchmark profile")
    schema = raw_payload.get("schema")
    if schema == PROFILE_SCHEMA:
        return dict(raw_payload), None
    if schema != PROFILE_OVERLAY_SCHEMA:
        raise ValueError(f"unsupported profile schema: {schema!r}")

    base_value = raw_payload.get("base_profile")
    base_sha256 = raw_payload.get("base_profile_sha256")
    if not isinstance(base_value, str) or not base_value:
        raise ValueError("v4 profile must declare base_profile")
    base_path = _resolve(source_path.parent, base_value, field="base_profile")
    _require_sha(base_path, base_sha256, label="base benchmark profile")
    base_payload = _load_json(base_path, label="base benchmark profile")
    if base_payload.get("schema") != PROFILE_SCHEMA:
        raise ValueError("v4 base profile must use the v3 complete-profile schema")

    overlay = {
        key: value
        for key, value in raw_payload.items()
        if key not in {"base_profile", "base_profile_sha256"}
    }
    payload = _deep_merge(base_payload, overlay)
    payload["schema"] = PROFILE_OVERLAY_SCHEMA
    return payload, {"path": str(base_path), "sha256": str(base_sha256)}


def _validate_benchmark_order(benchmarks: Any, *, label: str) -> None:
    if not isinstance(benchmarks, list) or any(
        not isinstance(item, dict) for item in benchmarks
    ):
        raise ValueError(f"{label} benchmarks must be a list of objects")
    order = tuple(item.get("id") for item in benchmarks)
    if order != BENCHMARK_ORDER:
        raise ValueError(f"{label} benchmark order differs from the sealed suite: {order}")
    for entry in benchmarks:
        _positive_int(
            entry.get("languages_expected"),
            field=f"{label}.{entry['id']}.languages_expected",
        )


def _load_release_manifest(
    payload: Mapping[str, Any], base: Path
) -> tuple[Path, Mapping[str, Any]]:
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("source must be an object")
    release_path = _resolve(
        base, source.get("release_manifest"), field="source.release_manifest"
    )
    _require_sha(
        release_path,
        source.get("release_manifest_sha256"),
        label="Stage-1 release manifest",
    )
    release = _load_json(release_path, label="Stage-1 release manifest")
    if release.get("release_name") != source.get("release_name"):
        raise ValueError("Stage-1 release-name drift")
    if release.get("status") != source.get("release_status"):
        raise ValueError("Stage-1 release-status drift")
    if int(release.get("independent_trajectories", 0)) != 64:
        raise ValueError("Stage-1 release no longer contains 64 trajectories")
    if release.get("data_dictionary_contract_status") != "pass":
        raise ValueError("Stage-1 release data-dictionary contract did not pass")
    return release_path, release


def load_campaign_profile(path: Path) -> CampaignProfile:
    source_path = path.resolve()
    raw_payload, base_profile = _load_profile_payload(source_path)
    if not isinstance(raw_payload.get("campaign_id"), str) or not raw_payload["campaign_id"]:
        raise ValueError("campaign_id must be a non-empty string")

    base = source_path.parent
    repo_root = _resolve(base, raw_payload.get("repo_root"), field="repo_root")
    harness_root = _resolve(base, raw_payload.get("harness_root"), field="harness_root")
    bridge_root = _resolve(base, raw_payload.get("bridge_root"), field="bridge_root")
    output_root = _resolve(base, raw_payload.get("output_root"), field="output_root")
    hf_home = _resolve(base, raw_payload.get("hf_home"), field="hf_home")
    lm_eval_python = _resolve(
        base, raw_payload.get("lm_eval_python"), field="lm_eval_python"
    )
    bridge_python = _resolve(
        base, raw_payload.get("bridge_python"), field="bridge_python"
    )
    for label, executable in (
        ("lm-eval interpreter", lm_eval_python),
        ("A40 Bridge interpreter", bridge_python),
    ):
        if not executable.is_file():
            raise FileNotFoundError(f"missing {label}: {executable}")
    if _git_head(harness_root) != raw_payload.get("harness_commit"):
        raise ValueError("lm-evaluation-harness commit drift")
    if _git_head(bridge_root) != raw_payload.get("bridge_commit"):
        raise ValueError("Megatron-Bridge commit drift")

    release_path, release = _load_release_manifest(raw_payload, base)

    tokenizer = raw_payload.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("tokenizer must be an object")
    tokenizer_model = _resolve(base, tokenizer.get("model"), field="tokenizer.model")
    _require_sha(tokenizer_model, tokenizer.get("sha256"), label="tokenizer")
    if int(tokenizer.get("vocab_size", 0)) != 64000:
        raise ValueError("unexpected tokenizer vocabulary size")

    request_sets = raw_payload.get("request_sets")
    if not isinstance(request_sets, dict):
        raise ValueError("request_sets must be an object")
    english = request_sets.get("english")
    multilingual = request_sets.get("multilingual")
    if not isinstance(english, dict) or not isinstance(multilingual, dict):
        raise ValueError("English and multilingual request-set contracts are required")
    _validate_benchmark_order(english.get("benchmarks"), label="English")
    if any(int(item["languages_expected"]) != 1 for item in english["benchmarks"]):
        raise ValueError("the English request set must contain one language per task")
    _validate_benchmark_order(multilingual.get("benchmarks"), label="Multilingual")
    for name, request_set in (("English", english), ("multilingual", multilingual)):
        sampling = request_set.get("sampling")
        if not isinstance(sampling, dict):
            raise ValueError(f"{name} sampling must be an object")
        for field in ("seed", "examples_per_language", "shared_population"):
            _positive_int(sampling.get(field), field=f"{name} sampling.{field}")

    lr_contract = raw_payload.get("learning_rate_contract")
    if not isinstance(lr_contract, dict) or lr_contract.get("name") != "jointmoe_x1":
        raise ValueError("missing JointMoE-x1 learning-rate contract")
    if lr_contract.get("formula") != "exp(8.39 - 0.81 * ln(N_act_nonembed) - 0.25 * ln(E))":
        raise ValueError("unexpected JointMoE-x1 formula")
    tolerance = float(lr_contract.get("relative_tolerance", -1))
    if tolerance <= 0:
        raise ValueError("learning-rate relative tolerance must be positive")

    resolved_models, source_artifacts = resolve_checkpoint_models(
        raw_payload, release, release_path
    )
    if base_profile is not None:
        source_artifacts["base_profile"] = base_profile
    payload = copy.deepcopy(raw_payload)
    payload["models"] = resolved_models
    payload["resolved_source_artifacts"] = source_artifacts

    selection = payload["selection"]
    primary = selection["canonical_release"]
    design_experts = _positive_int_list(primary.get("experts"), field="selection.experts")
    checkpoint_policy = selection.get("checkpoint_policy")
    if checkpoint_policy == "final_only":
        canonical_readouts_per_cell = {
            str(ladder): 1 for ladder in primary["ladders"]
        }
        replicated_readouts_per_cell = 1
    elif checkpoint_policy == "all_declared_readouts":
        schedules = primary.get("nominal_tokens_by_ladder")
        if not isinstance(schedules, dict):
            raise ValueError("all-readout selection is missing canonical schedules")
        canonical_readouts_per_cell = {
            str(ladder): len(
                _positive_int_list(
                    schedules[str(ladder)],
                    field=f"selection.{ladder}.nominal_tokens",
                )
            )
            for ladder in primary["ladders"]
        }
        replicated_readouts_per_cell = len(
            _positive_int_list(
                selection["b_seed_replication"].get("nominal_tokens"),
                field="selection.b_seed_replication.nominal_tokens",
            )
        )
    else:
        raise ValueError("unsupported checkpoint selection policy")
    expected_model_count = (
        sum(
            canonical_readouts_per_cell[str(ladder)]
            * len(primary["experts"])
            * len(primary["languages"])
            for ladder in primary["ladders"]
        )
        + len(selection["b_seed_replication"]["experts"])
        * len(selection["b_seed_replication"]["languages"])
        * len(selection["b_seed_replication"]["additional_seeds"])
        * replicated_readouts_per_cell
    )
    if len(resolved_models) != expected_model_count:
        raise ValueError("resolved campaign model count differs from declared cohorts")
    model_ids = tuple(item.get("id") for item in resolved_models)
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model IDs must be unique")
    observed_cells: set[tuple[str, int, int, int, int]] = set()
    for model in resolved_models:
        model_id = str(model["id"])
        ladder = str(model["ladder"])
        experts = _positive_int(model.get("experts"), field=f"{model_id}.experts")
        languages = _positive_int(model.get("languages"), field=f"{model_id}.languages")
        seed = _positive_int(model.get("seed"), field=f"{model_id}.seed")
        nominal_tokens = _positive_int(
            model.get("nominal_tokens"), field=f"{model_id}.nominal_tokens"
        )
        cell = (ladder, experts, languages, seed, nominal_tokens)
        if cell in observed_cells:
            raise ValueError(f"duplicate design cell: {cell}")
        observed_cells.add(cell)
        expected_request_set = "english" if languages == 1 else "multilingual"
        if model.get("request_set") != expected_request_set:
            raise ValueError(f"request-set policy mismatch for {model_id}")
        for field in (
            "active_parameters",
            "active_nonembedding_parameters",
            "checkpoint_step",
            "nominal_tokens",
            "tokens_seen",
            "configured_total_parameters",
            "dataset_shuffle_seed",
            "micro_batch_size",
            "expert_parallel_size",
            "readout_index",
            "readout_count",
        ):
            _positive_int(model.get(field), field=f"{model_id}.{field}")
        if int(model["readout_index"]) > int(model["readout_count"]):
            raise ValueError(f"invalid readout ordinal for {model_id}")
        if bool(model.get("is_final_readout")) != (
            int(model["readout_index"]) == int(model["readout_count"])
        ):
            raise ValueError(f"final-readout marker drift for {model_id}")
        for field in (
            "run_id",
            "training_manifest",
            "evaluation_manifest",
            "evaluation_spec",
            "trajectory_id",
            "readout_id",
            "recipe",
            "source_cohort",
            "replication_role",
        ):
            if not isinstance(model.get(field), str) or not model[field]:
                raise ValueError(f"{model_id}.{field} must be a non-empty string")
        active = int(model["active_nonembedding_parameters"])
        expected_lr = math.exp(
            8.39 - 0.81 * math.log(active) - 0.25 * math.log(experts)
        )
        if not math.isclose(
            float(model.get("peak_lr", -1)),
            expected_lr,
            rel_tol=tolerance,
            abs_tol=0.0,
        ):
            raise ValueError(f"JointMoE-x1 formula mismatch for {model_id}")

    runtime_estimation = payload.get("runtime_estimation")
    if (
        not isinstance(runtime_estimation, dict)
        or runtime_estimation.get("policy")
        != "largest_ladder_per_expert_and_request_set"
        or runtime_estimation.get("ladder") != "D"
        or float(runtime_estimation.get("safety_factor", 0)) < 1.0
    ):
        raise ValueError("runtime-estimation policy is invalid")
    calibration_models = [model for model in resolved_models if model["timing_calibration"]]
    expected_calibration_keys = {
        (experts, request_set)
        for experts in design_experts
        for request_set in ("english", "multilingual")
    }
    observed_calibration_keys = {
        (int(model["experts"]), str(model["request_set"]))
        for model in calibration_models
    }
    if observed_calibration_keys != expected_calibration_keys or len(calibration_models) != 8:
        raise ValueError("runtime calibration must contain one D model per E/request set")

    observed_world_sizes = {int(model["expert_parallel_size"]) for model in resolved_models}
    topology = payload.get("evaluation_topology_by_world_size")
    if not isinstance(topology, dict) or set(topology) != {
        str(value) for value in observed_world_sizes
    }:
        raise ValueError("evaluation topology must cover the source world sizes")
    for world_size in observed_world_sizes:
        item = topology[str(world_size)]
        if not isinstance(item, dict):
            raise ValueError(f"invalid topology for world size {world_size}")
        nodes = _positive_int(item.get("nodes"), field=f"world{world_size}.nodes")
        gpus_per_node = _positive_int(
            item.get("gpus_per_node"), field=f"world{world_size}.gpus_per_node"
        )
        if nodes * gpus_per_node != world_size:
            raise ValueError(f"A40 topology does not match world size {world_size}")
        if float(item.get("max_gpu_hours", 0)) <= 0:
            raise ValueError(f"world{world_size}.max_gpu_hours must be positive")
    override_policy = payload.get("a40_model_override_policy")
    if not isinstance(override_policy, dict) or set(override_policy) != {
        "base",
        "moe",
        "expert_parallel",
    }:
        raise ValueError("A40 model-override policy is incomplete")
    if any(not isinstance(value, dict) for value in override_policy.values()):
        raise ValueError("A40 model-override policy entries must be objects")
    budget_total = sum(
        float(topology[str(int(model["expert_parallel_size"]))]["max_gpu_hours"])
        for model in resolved_models
    )
    budget = payload.get("budget")
    if not isinstance(budget, dict) or budget.get("hardware") != "A40":
        raise ValueError("campaign budget must be denominated in A40 GPU-hours")
    if not math.isclose(
        budget_total, float(budget.get("max_gpu_hours", -1)), rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("per-model caps do not sum to the campaign budget")

    parity = payload.get("parity")
    if not isinstance(parity, dict):
        raise ValueError("parity must be an object")
    if float(parity.get("raw_nll_abs_tolerance", -1)) != RAW_NLL_ABS_TOLERANCE:
        raise ValueError("raw-NLL parity tolerance differs from scorer")
    if float(parity.get("nll_sum_relative_tolerance", -1)) != NLL_SUM_RELATIVE_TOLERANCE:
        raise ValueError("NLL-sum parity tolerance differs from scorer")

    profile = CampaignProfile(
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        payload=payload,
        repo_root=repo_root,
        harness_root=harness_root,
        bridge_root=bridge_root,
        output_root=output_root,
        hf_home=hf_home,
        lm_eval_python=lm_eval_python,
        bridge_python=bridge_python,
    )
    _validate_source_models(profile)
    return profile


def _validate_source_models(profile: CampaignProfile) -> None:
    import yaml

    tokenizer_sha = str(profile.payload["tokenizer"]["sha256"])
    lr_tolerance = float(profile.payload["learning_rate_contract"]["relative_tolerance"])
    for model in profile.payload["models"]:
        model_id = str(model["id"])
        experts = int(model["experts"])
        languages = int(model["languages"])
        checkpoint_step = int(model["checkpoint_step"])
        tokens_seen = int(model["tokens_seen"])
        topology = profile.topology(model)
        training_manifest_path, evaluation_spec_path, run_id = profile.training_paths(model)
        _require_sha(
            training_manifest_path,
            model.get("training_manifest_sha256"),
            label=f"{model_id} training manifest",
        )
        _require_sha(
            evaluation_spec_path,
            model.get("evaluation_spec_sha256"),
            label=f"{model_id} evaluation spec",
        )
        training_manifest = _load_json(
            training_manifest_path, label=f"{model_id} training manifest"
        )
        if training_manifest.get("status") != "complete":
            raise ValueError(f"training is not complete for {model_id}")
        if training_manifest.get("run_id") != run_id:
            raise ValueError(f"training run ID drift for {model_id}")
        metadata = training_manifest.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"missing training metadata for {model_id}")
        if int(metadata.get("expected_tokens", -1)) != tokens_seen:
            raise ValueError(f"realized-token drift for {model_id}")
        if int(metadata.get("final_checkpoint_step", -1)) != checkpoint_step:
            raise ValueError(f"final checkpoint-step drift for {model_id}")
        if int(metadata.get("micro_batch_size", -1)) != int(
            topology["micro_batch_size"]
        ):
            raise ValueError(f"evaluation MBS differs from training MBS for {model_id}")
        data_manifest = training_manifest.get("run_cfg", {}).get("data_manifest", "")
        if f"/tiers/K{languages}/" not in str(data_manifest):
            raise ValueError(f"training data tier drift for {model_id}")

        evaluation_spec = _load_json(
            evaluation_spec_path, label=f"{model_id} evaluation spec"
        )
        checkpoint = evaluation_spec.get("checkpoint")
        recipe = evaluation_spec.get("model_recipe")
        if not isinstance(checkpoint, dict) or not isinstance(recipe, dict):
            raise ValueError(f"invalid evaluation spec for {model_id}")
        if recipe.get("name") != model["recipe"]:
            raise ValueError(f"model recipe drift for {model_id}")
        if int(checkpoint.get("step", -1)) != checkpoint_step:
            raise ValueError(f"evaluation checkpoint-step drift for {model_id}")
        checkpoint_root = Path(str(checkpoint.get("path"))).resolve()
        if checkpoint_root != Path(str(metadata.get("final_checkpoint_root"))).resolve():
            raise ValueError(f"checkpoint-root drift for {model_id}")
        iteration_path = checkpoint_root / f"iter_{checkpoint_step:07d}"
        if not iteration_path.is_dir():
            raise FileNotFoundError(f"missing final checkpoint for {model_id}: {iteration_path}")
        checkpoint_tokenizer = iteration_path / "tokenizer" / "tokenizer.model"
        _require_sha(checkpoint_tokenizer, tokenizer_sha, label=f"{model_id} tokenizer")
        run_config_path = iteration_path / "run_config.yaml"
        if not run_config_path.is_file():
            raise FileNotFoundError(f"missing run config for {model_id}: {run_config_path}")
        run_config = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
        if not isinstance(run_config, dict):
            raise ValueError(f"invalid run config for {model_id}")
        actual_lr = float(run_config.get("optimizer", {}).get("lr", -1))
        expected_lr = float(model["peak_lr"])
        if not math.isclose(actual_lr, expected_lr, rel_tol=lr_tolerance, abs_tol=0.0):
            raise ValueError(
                f"LR-adjusted checkpoint mismatch for {model_id}: {actual_lr} != {expected_lr}"
            )
        if int(run_config.get("rng", {}).get("seed", -1)) != int(model["seed"]):
            raise ValueError(f"model-seed drift for {model_id}")
        if int(run_config.get("dataset", {}).get("random_seed", -1)) != int(
            model["dataset_shuffle_seed"]
        ):
            raise ValueError(f"dataset-seed drift for {model_id}")
        expected_model = recipe.get("expected_model")
        if not isinstance(expected_model, dict):
            raise ValueError(f"missing model contract for {model_id}")
        actual_experts = int(expected_model.get("num_moe_experts") or 1)
        if actual_experts != experts:
            raise ValueError(f"expert-count drift for {model_id}")
        actual_ep = int(expected_model.get("expert_model_parallel_size") or 1)
        if actual_ep != int(model["expert_parallel_size"]):
            raise ValueError(f"expert-parallel topology drift for {model_id}")
        if expected_model.get("moe_token_dispatcher_type") != model["moe_dispatcher"]:
            raise ValueError(f"MoE dispatcher drift for {model_id}")


__all__ = [
    "BENCHMARK_ORDER",
    "CampaignProfile",
    "load_campaign_profile",
]
