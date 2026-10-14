#!/usr/bin/env bash
set -euo pipefail

action="${1:-plan-calibration}"
case "${action}" in
  plan-calibration|submit-calibration|plan-remaining|submit-remaining|plan-all|submit-all) ;;
  *) echo "usage: $0 {plan-calibration|submit-calibration|plan-remaining|submit-remaining|plan-all|submit-all} [profile] [slurm-config]" >&2; exit 2 ;;
esac
operation="${action%%-*}"
cohort="${action#*-}"
if [[ "${cohort}" == "calibration" ]]; then
  calibration=true
else
  calibration=false
fi
[[ -z "${SLURM_JOB_ID:-}" ]] || {
  echo "refusing nested submission from Slurm job ${SLURM_JOB_ID}" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
profile="$(realpath -m "${repo_root}/${2:-configs/benchmarks/stage1_abcd_jointmoe_final_a40.json}")"
slurm_config="$(realpath -m "${repo_root}/${3:-configs/benchmarks/stage1_abcd_jointmoe_final_a40_slurm.json}")"
runner="${repo_root}/scripts/slurm/run_stage1_jointmoe_benchmark_job.sh"
jq_bin=/usr/bin/jq
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
bridge_root="$(realpath -m "${profile_dir}/$(${jq_bin} -er '.bridge_root' "${profile_contract}")")"
bridge_python="$(${jq_bin} -er '.bridge_python' "${profile_contract}")"
lm_eval_python="$(${jq_bin} -er '.lm_eval_python' "${profile_contract}")"
slurm_bin="$(${jq_bin} -er '.environment.slurm_bin' "${slurm_config}")"
preparation_receipt="${output_root}/preparation_receipt.json"
slurm_dir="${output_root}/slurm"
mkdir -p "${slurm_dir}"

PYTHONPATH="${repo_root}/src:${repo_root}" \
  "${lm_eval_python}" \
  -m multilingual_megatron.experiments.scaling.stage1_benchmarking.jointmoe \
  --profile "${profile}" validate >/dev/null
"${jq_bin}" -e '.schema == "stage1_jointmoe_benchmark_slurm/v3"' \
  "${slurm_config}" >/dev/null
for required in "${profile}" "${slurm_config}" "${runner}" "${preparation_receipt}"; do
  [[ -f "${required}" ]] || {
    echo "missing prepared campaign input: ${required}" >&2
    exit 3
  }
done
"${jq_bin}" -e \
  '.schema == "stage1_jointmoe_benchmark_preparation/v1" and .status == "validated_not_submitted" and .submitted == false and (.models | length) > 0' \
  "${preparation_receipt}" >/dev/null
profile_sha="$(sha256sum "${profile}" | awk '{print $1}')"
[[ "${profile_sha}" == "$(${jq_bin} -er '.profile.sha256' "${preparation_receipt}")" ]] || {
  echo "preparation receipt profile hash drift" >&2
  exit 3
}
runtime_estimate=""
if [[ "${cohort}" == "remaining" ]]; then
  runtime_estimate="${output_root}/runtime_estimate.json"
  [[ -f "${runtime_estimate}" ]] || {
    echo "missing calibration-based runtime estimate: ${runtime_estimate}" >&2
    exit 3
  }
  PYTHONPATH="${repo_root}/src:${repo_root}" \
    "${lm_eval_python}" \
    -m multilingual_megatron.experiments.scaling.stage1_benchmarking.jointmoe \
    --profile "${profile}" estimate-runtime >/dev/null
  campaign_id="$(${jq_bin} -er '.campaign_id' "${profile}")"
  "${jq_bin}" -e \
    --arg campaign_id "${campaign_id}" \
    --arg profile_sha256 "${profile_sha}" \
    '.schema == "stage1_jointmoe_benchmark_runtime_estimate/v1"
      and .status == "complete"
      and .campaign_id == $campaign_id
      and .profile_sha256 == $profile_sha256
      and .counts == {campaign_models:82, calibration_models:8, remaining_models:74}' \
    "${runtime_estimate}" >/dev/null
fi
if [[ "${cohort}" == "all" ]]; then
  mapfile -t model_ids < <(
    "${jq_bin}" -er '.models[].id' "${preparation_receipt}"
  )
else
  mapfile -t model_ids < <(
    "${jq_bin}" -er --argjson calibration "${calibration}" \
      '.models[] | select(.timing_calibration == $calibration) | .id' \
      "${preparation_receipt}"
  )
fi
[[ "${#model_ids[@]}" -gt 0 ]] || { echo "empty ${cohort} cohort" >&2; exit 3; }

module_init="$(${jq_bin} -er '.environment.module_init' "${slurm_config}")"
set +u
source "${module_init}"
set -u
module purge
mapfile -t gpu_modules < <("${jq_bin}" -er '.environment.gpu_modules[]' "${slurm_config}")
module load "${gpu_modules[@]}"

for model_id in "${model_ids[@]}"; do
  request_set="$(${jq_bin} -er --arg model_id "${model_id}" '.models[] | select(.id == $model_id) | .request_set' "${preparation_receipt}")"
  world_size="$(${jq_bin} -er --arg model_id "${model_id}" '.models[] | select(.id == $model_id) | .a40_topology.world_size' "${preparation_receipt}")"
  spec="${output_root}/specs/${request_set}/${model_id}.score_spec.json"
  [[ -f "${spec}" ]] || { echo "missing score spec: ${spec}" >&2; exit 3; }
  nodes="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].nodes' "${slurm_config}")"
  gpus_per_node="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].gpus_per_node' "${slurm_config}")"
  [[ "${nodes}" == "$(${jq_bin} -er --arg world_size "${world_size}" '.evaluation_topology_by_world_size[$world_size].nodes' "${profile_contract}")" ]] || {
    echo "node-count drift for ${model_id}" >&2
    exit 3
  }
  [[ "${gpus_per_node}" == "$(${jq_bin} -er --arg world_size "${world_size}" '.evaluation_topology_by_world_size[$world_size].gpus_per_node' "${profile_contract}")" ]] || {
    echo "GPU-count drift for ${model_id}" >&2
    exit 3
  }
  if [[ "${cohort}" != "all" ]]; then
    (
      export MASTER_PORT=29531
      export PROOF_PREFER_PYTHON_ENV_CUDA_LIBS=1
      export PROOF_ENABLE_BRIDGE_OPTIONAL_DEP_SHIMS=1
      export PROOF_REQUIRE_REAL_MODELOPT=1
      export PROOF_ENABLE_PYBIND11_SHIM=1
      export PYTHONPATH="${repo_root}/src:${repo_root}:${bridge_root}/src:${bridge_root}/3rdparty/Megatron-LM"
      "${bridge_python}" \
        -m multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_native_benchmark \
        --bridge-root "${bridge_root}" \
        --spec "${spec}" \
        --nnodes "${nodes}" \
        --nproc-per-node "${gpus_per_node}" \
        --python-executable "${bridge_python}" \
        --srun-arg=--mpi=pmix \
        --srun-arg=--cpu-bind=none \
        --srun-arg=--gpu-bind=none \
        --dry-run
    )
  fi
done

if [[ "${cohort}" == "all" ]]; then
  MASTER_ADDR=preflight.invalid MASTER_PORT=29531 \
    PYTHONPATH="${repo_root}/src:${repo_root}" \
    "${lm_eval_python}" \
    -m multilingual_megatron.experiments.scaling.stage1_benchmarking.jointmoe \
    --profile "${profile}" preflight >/dev/null
  preflight_receipt="${output_root}/preflight_receipt.json"
  "${jq_bin}" -e --argjson models "${#model_ids[@]}" \
    '.schema == "stage1_jointmoe_benchmark_preflight/v1"
      and .status == "validated_not_submitted"
      and .submitted == false
      and .counts.models == $models
      and .counts.request_manifests == 2' \
    "${preflight_receipt}" >/dev/null
fi

if [[ "${operation}" == "plan" ]]; then
  if [[ "${cohort}" == "all" ]]; then
    "${jq_bin}" -r \
      '.models
       | group_by([.ladder, .request_set, .a40_topology.world_size])[]
       | "ladder=\(.[0].ladder) request_set=\(.[0].request_set) world_size=\(.[0].a40_topology.world_size) jobs=\(length)"' \
      "${preparation_receipt}"
  else
    for model_id in "${model_ids[@]}"; do
      request_set="$(${jq_bin} -er --arg model_id "${model_id}" '.models[] | select(.id == $model_id) | .request_set' "${preparation_receipt}")"
      world_size="$(${jq_bin} -er --arg model_id "${model_id}" '.models[] | select(.id == $model_id) | .a40_topology.world_size' "${preparation_receipt}")"
      spec="${output_root}/specs/${request_set}/${model_id}.score_spec.json"
      nodes="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].nodes' "${slurm_config}")"
      gpus_per_node="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].gpus_per_node' "${slurm_config}")"
      printf '%s: LR=%s, step=%s, tokens=%s, requests=%s, %s A40 node(s) x %s GPU(s), cap=%s GPUh\n' \
        "${model_id}" \
        "$(${jq_bin} -er '.learning_rate_contract.peak_lr' "${spec}")" \
        "$(${jq_bin} -er '.checkpoint.step' "${spec}")" \
        "$(${jq_bin} -er '.checkpoint.tokens_seen' "${spec}")" \
        "${request_set}" "${nodes}" "${gpus_per_node}" \
        "$(${jq_bin} -er '.evaluation.max_gpu_hours' "${spec}")"
    done
  fi
  if [[ "${cohort}" == "all" ]]; then
    cohort_gpu_cap="$(${jq_bin} -er '[.models[].max_gpu_hours] | add' \
      "${preparation_receipt}")"
  else
    cohort_gpu_cap="$(${jq_bin} -er --argjson calibration "${calibration}" \
      '[.models[] | select(.timing_calibration == $calibration) | .max_gpu_hours] | add' \
      "${preparation_receipt}")"
  fi
  printf 'Prepared %s %s A40 jobs (%s GPUh maximum); no jobs submitted.\n' \
    "${#model_ids[@]}" "${cohort}" "${cohort_gpu_cap}"
  exit 0
fi

receipt="${slurm_dir}/submission_${cohort}_receipt.json"
jobs_tsv="${slurm_dir}/submitted_${cohort}_jobs.tsv"
jobs_json_path="${slurm_dir}/submitted_${cohort}_jobs.json"
[[ ! -e "${receipt}" && ! -e "${jobs_tsv}" && ! -e "${jobs_json_path}" ]] || {
  echo "submission receipt already exists; refusing duplicate submission" >&2
  exit 3
}
for model_id in "${model_ids[@]}"; do
  [[ ! -e "${output_root}/scores/${model_id}/scores.json" ]] || {
    echo "score output already exists for ${model_id}; refusing overwrite" >&2
    exit 3
  }
done

account="$(${jq_bin} -er '.account' "${slurm_config}")"
partition="$(${jq_bin} -er '.partition' "${slurm_config}")"
gpu_type="$(${jq_bin} -er '.gpu_type' "${slurm_config}")"
: > "${jobs_tsv}"
for model_id in "${model_ids[@]}"; do
  model_json="$(${jq_bin} -cer --arg model_id "${model_id}" '.models[] | select(.id == $model_id)' "${preparation_receipt}")"
  ladder="$(printf '%s' "${model_json}" | ${jq_bin} -er '.ladder')"
  experts="$(printf '%s' "${model_json}" | ${jq_bin} -er '.experts')"
  languages="$(printf '%s' "${model_json}" | ${jq_bin} -er '.languages')"
  seed="$(printf '%s' "${model_json}" | ${jq_bin} -er '.seed')"
  nominal_tokens="$(printf '%s' "${model_json}" | ${jq_bin} -er '.nominal_tokens')"
  readout_id="$(printf '%s' "${model_json}" | ${jq_bin} -er '.readout_id')"
  world_size="$(printf '%s' "${model_json}" | ${jq_bin} -er '.a40_topology.world_size')"
  nodes="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].nodes' "${slurm_config}")"
  gpus_per_node="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].gpus_per_node' "${slurm_config}")"
  qos="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].qos' "${slurm_config}")"
  exclude_nodes="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].exclude_nodes' "${slurm_config}")"
  job_id="$("${slurm_bin}/sbatch" --parsable --export=NIL --no-requeue \
    --partition="${partition}" --qos="${qos}" --account="${account}" \
    --nodes="${nodes}" --ntasks="${nodes}" --ntasks-per-node=1 \
    --cpus-per-task="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].cpus_per_task' "${slurm_config}")" \
    --gres="gpu:${gpu_type}:${gpus_per_node}" \
    --mem="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].memory_per_node' "${slurm_config}")" \
    --time="$(${jq_bin} -er --arg world_size "${world_size}" '.job_classes[$world_size].time' "${slurm_config}")" \
    --exclude="${exclude_nodes}" \
    --job-name="bm-${ladder,,}-e${experts}-k${languages}-d$((nominal_tokens / 1000000))m-s${seed}" \
    --output="${slurm_dir}/%x-%j.out" \
    "${runner}" "${model_id}" "${profile#${repo_root}/}" "${slurm_config#${repo_root}/}")"
  job_id="${job_id%%;*}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${model_id}" "${job_id}" "${ladder}" "${experts}" "${languages}" "${seed}" \
    "${nominal_tokens}" "${readout_id}" "${world_size}" "${nodes}" "${gpus_per_node}" \
    >> "${jobs_tsv}"
done

"${jq_bin}" -Rn \
  '[inputs | split("\t") | {key:.[0], value:{job_id:.[1],ladder:.[2],experts:(.[3]|tonumber),languages:(.[4]|tonumber),seed:(.[5]|tonumber),nominal_tokens:(.[6]|tonumber),readout_id:.[7],world_size:(.[8]|tonumber),nodes:(.[9]|tonumber),gpus_per_node:(.[10]|tonumber),dependency:null}}] | from_entries' \
  < "${jobs_tsv}" > "${jobs_json_path}"
runtime_estimate_json=null
if [[ -n "${runtime_estimate}" ]]; then
  runtime_estimate_json="$(${jq_bin} -n \
    --arg path "${runtime_estimate}" \
    --arg sha256 "$(sha256sum "${runtime_estimate}" | awk '{print $1}')" \
    '{path:$path,sha256:$sha256}')"
fi
preflight_receipt_json=null
if [[ "${cohort}" == "all" ]]; then
  preflight_receipt_json="$(${jq_bin} -n \
    --arg path "${preflight_receipt}" \
    --arg sha256 "$(sha256sum "${preflight_receipt}" | awk '{print $1}')" \
    '{path:$path,sha256:$sha256}')"
fi
"${jq_bin}" -n \
  --arg submitted_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg repo_head "$(git rev-parse HEAD)" \
  --arg bridge_head "$(git -C "${bridge_root}" rev-parse HEAD)" \
  --arg account "${account}" \
  --arg partition "${partition}" \
  --arg cohort "${cohort}" \
  --arg profile "${profile}" \
  --arg profile_sha256 "${profile_sha}" \
  --arg preparation_receipt "${preparation_receipt}" \
  --arg preparation_receipt_sha256 "$(sha256sum "${preparation_receipt}" | awk '{print $1}')" \
  --arg jobs_tsv "${jobs_tsv}" \
  --arg jobs_tsv_sha256 "$(sha256sum "${jobs_tsv}" | awk '{print $1}')" \
  --arg jobs_json_path "${jobs_json_path}" \
  --arg jobs_json_sha256 "$(sha256sum "${jobs_json_path}" | awk '{print $1}')" \
  --slurpfile jobs "${jobs_json_path}" \
  --argjson runtime_estimate "${runtime_estimate_json}" \
  --argjson preflight_receipt "${preflight_receipt_json}" \
  '{schema:"stage1_jointmoe_benchmark_submission/v1",cohort:$cohort,submitted_at:$submitted_at,repo_head:$repo_head,bridge_head:$bridge_head,account:$account,partition:$partition,hardware:"A40",profile:{path:$profile,sha256:$profile_sha256},preparation_receipt:{path:$preparation_receipt,sha256:$preparation_receipt_sha256},preflight_receipt:$preflight_receipt,runtime_estimate:$runtime_estimate,jobs_tsv:{path:$jobs_tsv,sha256:$jobs_tsv_sha256},jobs_json:{path:$jobs_json_path,sha256:$jobs_json_sha256},jobs:$jobs[0],no_cpu_dependency_jobs:true}' \
  > "${receipt}"
printf 'Submitted %s %s A40 benchmark jobs; receipt=%s\n' \
  "${#model_ids[@]}" "${cohort}" "${receipt}"
"${jq_bin}" -r '.jobs | to_entries[] | "\(.key)=\(.value.job_id)"' "${receipt}"
