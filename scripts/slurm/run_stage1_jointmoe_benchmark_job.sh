#!/usr/bin/env bash
set -euo pipefail

model_id="${1:-}"
profile_arg="${2:-configs/benchmarks/stage1_abcd_jointmoe_final_a40.json}"
slurm_arg="${3:-configs/benchmarks/stage1_abcd_jointmoe_final_a40_slurm.json}"
[[ -n "${model_id}" ]] || {
  echo "usage: $0 MODEL_ID [profile] [slurm-config]" >&2
  exit 2
}
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "runner requires a Slurm allocation" >&2
  exit 2
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src" ]]; then
  repo_root="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${repo_root}"
jq_bin=/usr/bin/jq
profile="$(realpath -m "${repo_root}/${profile_arg}")"
slurm_config="$(realpath -m "${repo_root}/${slurm_arg}")"
profile_dir="$(dirname "${profile}")"
profile_contract="${profile}"
base_profile_relative="$(${jq_bin} -er '.base_profile // ""' "${profile}")"
if [[ -n "${base_profile_relative}" ]]; then
  profile_contract="$(realpath -m "${profile_dir}/${base_profile_relative}")"
  [[ -f "${profile_contract}" ]] || {
    echo "missing base campaign profile: ${profile_contract}" >&2
    exit 3
  }
  [[ "$(sha256sum "${profile_contract}" | awk '{print $1}')" == "$(${jq_bin} -er '.base_profile_sha256' "${profile}")" ]] || {
    echo "base campaign profile hash drift" >&2
    exit 3
  }
fi
output_root="$(realpath -m "${profile_dir}/$(${jq_bin} -er '.output_root' "${profile}")")"
preparation_receipt="${output_root}/preparation_receipt.json"
[[ -f "${preparation_receipt}" ]] || {
  echo "missing preparation receipt: ${preparation_receipt}" >&2
  exit 3
}

model_json="$(${jq_bin} -cer --arg model_id "${model_id}" \
  '.models[] | select(.id == $model_id)' "${preparation_receipt}")" || {
  echo "model is absent from campaign profile: ${model_id}" >&2
  exit 3
}
experts="$(printf '%s' "${model_json}" | ${jq_bin} -er '.experts')"
languages="$(printf '%s' "${model_json}" | ${jq_bin} -er '.languages')"
ladder="$(printf '%s' "${model_json}" | ${jq_bin} -er '.ladder')"
seed="$(printf '%s' "${model_json}" | ${jq_bin} -er '.seed')"
nominal_tokens="$(printf '%s' "${model_json}" | ${jq_bin} -er '.nominal_tokens')"
readout_id="$(printf '%s' "${model_json}" | ${jq_bin} -er '.readout_id')"
readout_index="$(printf '%s' "${model_json}" | ${jq_bin} -er '.readout_index')"
readout_count="$(printf '%s' "${model_json}" | ${jq_bin} -er '.readout_count')"
world_size="$(printf '%s' "${model_json}" | ${jq_bin} -er '.a40_topology.world_size')"
request_set="$(printf '%s' "${model_json}" | ${jq_bin} -er '.request_set')"

bridge_root="$(realpath -m "${profile_dir}/$(${jq_bin} -er '.bridge_root' "${profile_contract}")")"
harness_root="$(realpath -m "${profile_dir}/$(${jq_bin} -er '.harness_root' "${profile_contract}")")"
bridge_python="$(${jq_bin} -er '.bridge_python' "${profile_contract}")"
lm_eval_python="$(${jq_bin} -er '.lm_eval_python' "${profile_contract}")"
hf_home="$(${jq_bin} -er '.hf_home' "${profile_contract}")"
spec="${output_root}/specs/${request_set}/${model_id}.score_spec.json"
slurm_bin="$(${jq_bin} -er '.environment.slurm_bin' "${slurm_config}")"
module_init="$(${jq_bin} -er '.environment.module_init' "${slurm_config}")"
expected_nodes="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].nodes' "${slurm_config}")"
gpus_per_node="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].gpus_per_node' "${slurm_config}")"

for required in "${profile}" "${slurm_config}" "${preparation_receipt}" "${spec}"; do
  [[ -f "${required}" ]] || {
    echo "missing sealed input: ${required}" >&2
    exit 3
  }
done
"${jq_bin}" -e '.schema == "stage1_jointmoe_benchmark_slurm/v3"' \
  "${slurm_config}" >/dev/null
request_manifest="$(${jq_bin} -er '.request_manifest.path' "${spec}")"
training_manifest="$(${jq_bin} -er '.source_training_manifest.path' "${spec}")"
for required in "${request_manifest}" "${training_manifest}"; do
  [[ -f "${required}" ]] || {
    echo "missing sealed source: ${required}" >&2
    exit 3
  }
done
[[ "$(git -C "${bridge_root}" rev-parse HEAD)" == "$(${jq_bin} -er '.bridge_commit' "${profile_contract}")" ]] || {
  echo "Megatron-Bridge commit drift" >&2
  exit 3
}
expected_world_size="$(${jq_bin} -er '.evaluation.expected_world_size' "${spec}")"
profile_nodes="$(${jq_bin} -er --arg world_size "${world_size}" '.evaluation_topology_by_world_size[$world_size].nodes' "${profile_contract}")"
profile_gpus_per_node="$(${jq_bin} -er --arg world_size "${world_size}" '.evaluation_topology_by_world_size[$world_size].gpus_per_node' "${profile_contract}")"
[[ "${expected_nodes}" == "${profile_nodes}" && "${gpus_per_node}" == "${profile_gpus_per_node}" ]] || {
  echo "Slurm topology differs from campaign profile for E=${experts}" >&2
  exit 3
}
[[ "$((expected_nodes * gpus_per_node))" == "${expected_world_size}" ]] || {
  echo "Slurm topology differs from score-spec world size" >&2
  exit 3
}
[[ "${SLURM_NNODES:-0}" == "${expected_nodes}" ]] || {
  echo "allocated nodes ${SLURM_NNODES:-0} != expected ${expected_nodes}" >&2
  exit 3
}
profile_sha="$(sha256sum "${profile}" | awk '{print $1}')"
request_sha="$(sha256sum "${request_manifest}" | awk '{print $1}')"
training_sha="$(sha256sum "${training_manifest}" | awk '{print $1}')"
[[ "${profile_sha}" == "$(${jq_bin} -er '.profile.sha256' "${spec}")" ]] || {
  echo "score-spec profile hash drift" >&2
  exit 3
}
[[ "${profile_sha}" == "$(${jq_bin} -er '.profile.sha256' "${preparation_receipt}")" ]] || {
  echo "preparation-receipt profile hash drift" >&2
  exit 3
}
preflight_receipt="${output_root}/preflight_receipt.json"
if [[ -n "${base_profile_relative}" && ! -f "${preflight_receipt}" ]]; then
  echo "missing full-campaign preflight receipt: ${preflight_receipt}" >&2
  exit 3
fi
if [[ -f "${preflight_receipt}" ]]; then
  "${jq_bin}" -e --arg profile_sha256 "${profile_sha}" --arg model_id "${model_id}" \
    '.schema == "stage1_jointmoe_benchmark_preflight/v1"
      and .status == "validated_not_submitted"
      and .profile.sha256 == $profile_sha256
      and any(.models[]; .model_id == $model_id)' \
    "${preflight_receipt}" >/dev/null
  while IFS=$'\t' read -r source_relative source_path source_sha256; do
    [[ "${source_path}" == "${repo_root}/${source_relative}" ]] || {
      echo "preflight runtime-source path drift: ${source_relative}" >&2
      exit 3
    }
    [[ -f "${source_path}" ]] || {
      echo "missing preflight runtime source: ${source_path}" >&2
      exit 3
    }
    [[ "$(sha256sum "${source_path}" | awk '{print $1}')" == "${source_sha256}" ]] || {
      echo "preflight runtime-source hash drift: ${source_relative}" >&2
      exit 3
    }
  done < <("${jq_bin}" -r \
    '.runtime_sources | to_entries[] | [.key, .value.path, .value.sha256] | @tsv' \
    "${preflight_receipt}")
fi
[[ "${request_sha}" == "$(${jq_bin} -er '.request_manifest.sha256' "${spec}")" ]] || {
  echo "score-spec request-manifest hash drift" >&2
  exit 3
}
[[ "${training_sha}" == "$(${jq_bin} -er '.source_training_manifest.sha256' "${spec}")" ]] || {
  echo "score-spec training-manifest hash drift" >&2
  exit 3
}

job_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
job_started_epoch="$(date +%s)"
record_dir="${output_root}/slurm/jobs/${SLURM_JOB_ID}-${model_id}"
runtime_receipt="${output_root}/runtime/${model_id}.json"
[[ ! -e "${runtime_receipt}" ]] || {
  echo "runtime receipt already exists for ${model_id}; refusing overwrite" >&2
  exit 3
}
mkdir -p "${record_dir}"
"${slurm_bin}/scontrol" show job "${SLURM_JOB_ID}" -o > "${record_dir}/scontrol_job.txt"
"${slurm_bin}/scontrol" show hostnames "${SLURM_JOB_NODELIST}" > "${record_dir}/selected_nodes.txt"
git rev-parse HEAD > "${record_dir}/repo_head.txt"
git -C "${bridge_root}" rev-parse HEAD > "${record_dir}/bridge_head.txt"
sha256sum \
  "${profile}" "${slurm_config}" "${preparation_receipt}" "${spec}" \
  "${request_manifest}" "${training_manifest}" \
  "${repo_root}/scripts/slurm/run_stage1_jointmoe_benchmark_job.sh" \
  "${repo_root}/src/multilingual_megatron/experiments/scaling/stage1_benchmarking/jointmoe.py" \
  "${repo_root}/src/multilingual_megatron/experiments/scaling/stage1_benchmarking/api.py" \
  "${repo_root}/src/multilingual_megatron/experiments/scaling/stage1_benchmarking/campaign_profile.py" \
  "${repo_root}/src/multilingual_megatron/experiments/scaling/stage1_benchmarking/campaign_models.py" \
  "${repo_root}/src/multilingual_megatron/experiments/scaling/stage1_benchmarking/contracts.py" \
  "${repo_root}/src/multilingual_megatron/experiments/scaling/stage1_benchmarking/replay.py" \
  "${repo_root}/src/multilingual_megatron/experiments/scaling/stage1_benchmarking/tasks.py" \
  "${repo_root}/src/multilingual_megatron/pipeline/backends/megatron_bridge/run_megatron_bridge_native_benchmark.py" \
  "${repo_root}/src/multilingual_megatron/pipeline/backends/megatron_bridge/run_study_bridge_recipe.py" \
  "${repo_root}/src/multilingual_megatron/pipeline/stages/benchmarking/native_megatron_lm_eval.py" \
  "${repo_root}/src/multilingual_megatron/pipeline/stages/benchmarking/native_megatron_runner.py" \
  "${repo_root}/src/multilingual_megatron/pipeline/stages/benchmarking/native_megatron_scoring.py" \
  > "${record_dir}/input_sha256.txt"
"${jq_bin}" -n \
  --arg job_id "${SLURM_JOB_ID}" \
  --arg model_id "${model_id}" \
  --arg ladder "${ladder}" \
  --arg request_set "${request_set}" \
  --arg nodes "${SLURM_JOB_NODELIST}" \
  --argjson experts "${experts}" \
  --argjson languages "${languages}" \
  --argjson seed "${seed}" \
  --argjson nominal_tokens "${nominal_tokens}" \
  --arg readout_id "${readout_id}" \
  --argjson readout_index "${readout_index}" \
  --argjson readout_count "${readout_count}" \
  --argjson expected_nodes "${expected_nodes}" \
  --argjson gpus_per_node "${gpus_per_node}" \
  --argjson world_size "${expected_world_size}" \
  --arg started_at "${job_started_at}" \
  '{schema:"stage1_jointmoe_benchmark_scheduler_context/v1",holder_job_id:$job_id,model_id:$model_id,trajectory_readout:{readout_id:$readout_id,nominal_tokens:$nominal_tokens,index:$readout_index,count:$readout_count},ladder:$ladder,experts:$experts,languages:$languages,seed:$seed,request_set:$request_set,allocated_nodes:$nodes,node_subset:$nodes,expected_nodes:$expected_nodes,gpus_per_node:$gpus_per_node,world_size:$world_size,started_at:$started_at}' \
  > "${record_dir}/scheduler_context.json"

set +u
source "${module_init}"
set -u
module purge
mapfile -t gpu_modules < <("${jq_bin}" -er '.environment.gpu_modules[]' "${slurm_config}")
module load "${gpu_modules[@]}"
export PATH="${slurm_bin}:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export HF_HOME="${hf_home}"
export HF_DATASETS_CACHE="${hf_home}/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=1
export BRIDGE_PREFER_VENV_CUDA_LIBS=0
export PROOF_PREFER_PYTHON_ENV_CUDA_LIBS=1
export PROOF_ENABLE_PYBIND11_SHIM=1
export PROOF_ENABLE_BRIDGE_OPTIONAL_DEP_SHIMS=1
export PROOF_REQUIRE_REAL_MODELOPT=1
export PROOF_DATASET_BUILDER_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"
unset MASTER_ADDR
export CONDA_PREFIX="$(dirname "$(dirname "${bridge_python}")")"
export CONDA_DEFAULT_ENV="$(basename "${CONDA_PREFIX}")"
export SLURM_EXPORT_ENV=ALL
export PYTHONPATH="${repo_root}/src:${repo_root}:${bridge_root}/src:${bridge_root}/3rdparty/Megatron-LM"
export TMPDIR="$("${jq_bin}" -er '.environment.tmp_root' "${slurm_config}")"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export TRITON_CACHE_DIR="${output_root}/runtime_cache/triton/${SLURM_JOB_ID}"
export TRITON_HOME="${output_root}/runtime_cache/triton_home/${SLURM_JOB_ID}"
mkdir -p "${TRITON_CACHE_DIR}" "${TRITON_HOME}"

actual_local_gpus="$(${bridge_python} -c 'import torch; print(torch.cuda.device_count())')"
[[ "${actual_local_gpus}" == "${gpus_per_node}" ]] || {
  echo "visible local GPU count ${actual_local_gpus} != expected ${gpus_per_node}" >&2
  exit 4
}
mapfile -t srun_args < <("${jq_bin}" -er --arg world_size "${world_size}" '.job_classes[$world_size].srun_args[]' "${slurm_config}")
"${slurm_bin}/srun" --nodes="${expected_nodes}" --ntasks="${expected_nodes}" \
  --ntasks-per-node=1 "${srun_args[@]}" --label /usr/bin/hostname \
  > "${record_dir}/task_hosts.txt"
"${slurm_bin}/srun" --nodes="${expected_nodes}" --ntasks="${expected_nodes}" \
  --ntasks-per-node=1 "${srun_args[@]}" --label /bin/bash -c \
  'nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader' \
  > "${record_dir}/gpu_inventory.csv"

launcher_srun_args=()
for value in "${srun_args[@]}"; do
  launcher_srun_args+=("--srun-arg=${value}")
done
"${bridge_python}" \
  -m multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_native_benchmark \
  --bridge-root "${bridge_root}" \
  --spec "${spec}" \
  --nnodes "${expected_nodes}" \
  --nproc-per-node "${gpus_per_node}" \
  --python-executable "${bridge_python}" \
  "${launcher_srun_args[@]}" \
  --preflight-cuda
"${jq_bin}" -e \
  '.schema == "megatron_bridge_native_benchmark_launcher_result/v1" and .status == "complete" and .return_code == 0' \
  "${output_root}/scores/${model_id}/launcher_result.json" >/dev/null
"${jq_bin}" -e \
  '.schema == "stage1_native_megatron_benchmark_scores/v1" and .status == "complete"' \
  "${output_root}/scores/${model_id}/scores.json" >/dev/null
printf 'ok\n' > "${record_dir}/.score_complete"

export PYTHONPATH="${repo_root}/src:${harness_root}:${repo_root}"
"${lm_eval_python}" \
  -m multilingual_megatron.experiments.scaling.stage1_benchmarking.jointmoe \
  --profile "${profile}" aggregate --model-id "${model_id}"
"${jq_bin}" -e \
  '.schema == "stage1_lm_eval_benchmark_metrics/v1" and .status == "complete"' \
  "${output_root}/metrics/${model_id}/metrics.json" >/dev/null
printf 'ok\n' > "${record_dir}/.aggregate_complete"
job_completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
job_completed_epoch="$(date +%s)"
job_wall_seconds="$((job_completed_epoch - job_started_epoch))"
scores_path="${output_root}/scores/${model_id}/scores.json"
metrics_path="${output_root}/metrics/${model_id}/metrics.json"
mkdir -p "$(dirname "${runtime_receipt}")"
runtime_tmp="${runtime_receipt}.tmp-${SLURM_JOB_ID}"
"${jq_bin}" -n \
  --arg campaign_id "$(${jq_bin} -er '.campaign_id' "${profile}")" \
  --arg model_id "${model_id}" \
  --arg ladder "${ladder}" \
  --arg request_set "${request_set}" \
  --arg profile "${profile}" \
  --arg profile_sha256 "${profile_sha}" \
  --arg score_path "${scores_path}" \
  --arg score_sha256 "$(sha256sum "${scores_path}" | awk '{print $1}')" \
  --arg score_result_sha256 "$(${jq_bin} -er '.result_sha256' "${scores_path}")" \
  --arg metrics_path "${metrics_path}" \
  --arg metrics_sha256 "$(sha256sum "${metrics_path}" | awk '{print $1}')" \
  --arg slurm_job_id "${SLURM_JOB_ID}" \
  --arg started_at "${job_started_at}" \
  --arg completed_at "${job_completed_at}" \
  --argjson experts "${experts}" \
  --argjson languages "${languages}" \
  --argjson seed "${seed}" \
  --argjson nominal_tokens "${nominal_tokens}" \
  --arg readout_id "${readout_id}" \
  --argjson readout_index "${readout_index}" \
  --argjson readout_count "${readout_count}" \
  --argjson job_wall_seconds "${job_wall_seconds}" \
  --argjson gpu_count "${expected_world_size}" \
  --argjson scorer_timing "$(${jq_bin} -c '.timing' "${scores_path}")" \
  '{schema:"stage1_jointmoe_benchmark_runtime/v1",status:"complete",campaign_id:$campaign_id,model_id:$model_id,trajectory_readout:{readout_id:$readout_id,nominal_tokens:$nominal_tokens,index:$readout_index,count:$readout_count},ladder:$ladder,experts:$experts,languages:$languages,seed:$seed,request_set:$request_set,profile:{path:$profile,sha256:$profile_sha256},score_result:{path:$score_path,sha256:$score_sha256,result_sha256:$score_result_sha256},metrics:{path:$metrics_path,sha256:$metrics_sha256},slurm:{job_id:$slurm_job_id},timing:{started_at:$started_at,completed_at:$completed_at,job_wall_seconds:$job_wall_seconds,gpu_count:$gpu_count,gpu_hours:($job_wall_seconds * $gpu_count / 3600),scorer:$scorer_timing}}' \
  > "${runtime_tmp}"
mv "${runtime_tmp}" "${runtime_receipt}"
printf 'ok\n' > "${record_dir}/.job_complete"
