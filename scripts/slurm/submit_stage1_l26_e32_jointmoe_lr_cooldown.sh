#!/usr/bin/env bash
# Source-locked, dependency-gated launcher for the final Stage-1 L26/E32 cooldown.

set -euo pipefail

resolve_repo_root() {
  local candidate
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    candidate="${SLURM_SUBMIT_DIR:-}"
    [[ -n "${candidate}" ]] || {
      echo "[COOLDOWN_LAUNCH_ERROR] SLURM_SUBMIT_DIR is unavailable in the batch job" >&2
      exit 2
    }
  else
    candidate="$(dirname "${BASH_SOURCE[0]}")/../.."
  fi
  (cd "${candidate}" && pwd -P)
}

repo_root="$(resolve_repo_root)"
cd "${repo_root}"

readonly script_path="${repo_root}/scripts/slurm/submit_stage1_l26_e32_jointmoe_lr_cooldown.sh"
readonly launch_contract_relative="configs/stage1/production/stage1_l26_e32_k200_jointmoe_lr_cooldown_launch_v1.json"
readonly launch_contract="${repo_root}/${launch_contract_relative}"
readonly expected_launch_contract_sha256="5fec2ade45cea33f93dec57df3d54b49bdc0abfea95346a4a7225844875e510c"
readonly runtime_python="/opt/conda/envs/bridge-train/bin/python"
readonly conda_base="/opt/software/cluster/EB-SW/software/Miniforge3/25.3.0-3"
readonly module_init="/etc/profile.d/z00_lmod.sh"
readonly slurm_bin="/opt/software/slurm/25.11.5-1/bin"
readonly output_root="${repo_root}/outputs/pipeline_runs/stage1_l26_e32_k200_jointmoe_lr_production_v1"
readonly cooldown_root="${output_root}/cooldown"
readonly receipt_root="${output_root}/preflight"
readonly source_snapshot_store="${receipt_root}/source_snapshots"
readonly attempt_id="20260907-jointmoe-lr-cooldown-retry2-verified-resume-v1"
readonly slurm_log_root="${cooldown_root}/logs/attempts/${attempt_id}/slurm"
readonly submission_guard="${receipt_root}/cooldown_submission_retry2_once.lock"
SOURCE_SNAPSHOT_ROOT=""
TRUNK_TERMINAL=""

usage() {
  cat <<'EOF'
usage: submit_stage1_l26_e32_jointmoe_lr_cooldown.sh {plan|submit|run}

plan    Verify the prepared source lock and print the exact safe command.
submit  Verify the completed trunk, snapshot committed source, then submit once.
run     Internal Slurm entry point; do not invoke manually.
EOF
}

die() {
  echo "[COOLDOWN_LAUNCH_ERROR] $*" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

tracked_tree_sha256() {
  local path="$1"
  git ls-files -- "${path}" \
    | LC_ALL=C sort \
    | while IFS= read -r file; do
        case "${file}" in
          *.py|*.json|*.yaml|*.yml) sha256sum "${file}" ;;
        esac
      done \
    | sha256sum \
    | awk '{print $1}'
}

filesystem_tree_sha256() {
  local root="$1"
  local path="$2"
  (
    cd "${root}"
    find "${path}" -type f \( \
      -name '*.py' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
    \) -print \
      | LC_ALL=C sort \
      | while IFS= read -r file; do sha256sum "${file}"; done \
      | sha256sum \
      | awk '{print $1}'
  )
}

verify_submission_snapshot() {
  local snapshot_root="$1"
  local source_commit="$2"
  local snapshot_contract="${snapshot_root}/${launch_contract_relative}"
  local snapshot_manifest="${snapshot_root}/submission_source_manifest.json"
  [[ -f "${snapshot_contract}" ]] || die "submission snapshot lacks launch contract"
  [[ -f "${snapshot_manifest}" ]] || die "submission snapshot lacks source manifest"
  [[ -L "${snapshot_root}/outputs" ]] || die "submission snapshot lacks outputs link"
  [[ "$(readlink -f "${snapshot_root}/outputs")" == "${repo_root}/outputs" ]] \
    || die "submission snapshot outputs link is invalid"

  jq -e \
    --arg source_commit "${source_commit}" \
    --arg bridge_commit "$(jq -r '.source_lock.bridge_commit' "${snapshot_contract}")" \
    --arg megatron_lm_commit "$(jq -r '.source_lock.megatron_lm_commit' "${snapshot_contract}")" \
    '.schema == "stage1_l26_e32_submission_source_snapshot/v1"
      and .source_commit == $source_commit
      and .bridge_commit == $bridge_commit
      and .megatron_lm_commit == $megatron_lm_commit' \
    "${snapshot_manifest}" >/dev/null \
    || die "submission snapshot manifest mismatch"

  [[ "$(sha256_file "${snapshot_contract}")" == "${expected_launch_contract_sha256}" ]] \
    || die "submission snapshot launch contract hash mismatch"

  local relative expected actual
  while IFS=$'\t' read -r relative expected; do
    [[ -f "${snapshot_root}/${relative}" ]] \
      || die "submission snapshot lacks locked file: ${relative}"
    actual="$(sha256_file "${snapshot_root}/${relative}")"
    [[ "${actual}" == "${expected}" ]] \
      || die "submission snapshot source-lock mismatch: ${relative}"
  done < <(jq -r '.source_lock.files[] | [.path, .sha256] | @tsv' "${snapshot_contract}")

  expected="$(jq -r '.source_lock.pipeline_python_tree_sha256' "${snapshot_contract}")"
  actual="$(filesystem_tree_sha256 "${snapshot_root}" src/multilingual_megatron/pipeline)"
  [[ "${actual}" == "${expected}" ]] || die "submission snapshot pipeline hash mismatch"

  expected="$(jq -r '.source_lock.pipeline_config_tree_sha256' "${snapshot_contract}")"
  actual="$(filesystem_tree_sha256 "${snapshot_root}" configs/pipeline)"
  [[ "${actual}" == "${expected}" ]] || die "submission snapshot config hash mismatch"
}

materialize_submission_snapshot() {
  local source_commit="$1"
  local snapshot_root="${source_snapshot_store}/${source_commit}"
  if [[ -d "${snapshot_root}" ]]; then
    verify_submission_snapshot "${snapshot_root}" "${source_commit}"
    SOURCE_SNAPSHOT_ROOT="${snapshot_root}"
    return
  fi
  [[ ! -e "${snapshot_root}" ]] || die "invalid submission snapshot path: ${snapshot_root}"

  command -v tar >/dev/null || die "tar is unavailable"
  git cat-file -e "${source_commit}^{commit}" \
    || die "submission commit is unavailable: ${source_commit}"

  mkdir -p "${source_snapshot_store}"
  local temporary
  temporary="$(mktemp -d "${source_snapshot_store}/.${source_commit}.tmp.XXXXXX")"
  git archive --format=tar "${source_commit}" -- src configs \
    | tar -xf - -C "${temporary}"

  local snapshot_contract bridge_commit megatron_lm_commit
  snapshot_contract="${temporary}/${launch_contract_relative}"
  [[ "$(sha256_file "${snapshot_contract}")" == "${expected_launch_contract_sha256}" ]] \
    || die "archived launch contract hash mismatch"
  bridge_commit="$(jq -r '.source_lock.bridge_commit' "${snapshot_contract}")"
  megatron_lm_commit="$(jq -r '.source_lock.megatron_lm_commit' "${snapshot_contract}")"

  [[ "$(git ls-tree "${source_commit}" external/Megatron-Bridge | awk '{print $3}')" == "${bridge_commit}" ]] \
    || die "submission commit Bridge gitlink mismatch"
  git -C "${repo_root}/external/Megatron-Bridge" cat-file -e "${bridge_commit}^{commit}" \
    || die "sealed Bridge commit is unavailable"
  [[ "$(git -C "${repo_root}/external/Megatron-Bridge" ls-tree "${bridge_commit}" 3rdparty/Megatron-LM | awk '{print $3}')" == "${megatron_lm_commit}" ]] \
    || die "sealed Bridge commit Megatron-LM gitlink mismatch"
  git -C "${repo_root}/external/Megatron-Bridge/3rdparty/Megatron-LM" \
    cat-file -e "${megatron_lm_commit}^{commit}" \
    || die "sealed Megatron-LM commit is unavailable"

  mkdir -p "${temporary}/external/Megatron-Bridge"
  git -C "${repo_root}/external/Megatron-Bridge" \
    archive --format=tar "${bridge_commit}" \
    | tar -xf - -C "${temporary}/external/Megatron-Bridge"
  mkdir -p "${temporary}/external/Megatron-Bridge/3rdparty/Megatron-LM"
  git -C "${repo_root}/external/Megatron-Bridge/3rdparty/Megatron-LM" \
    archive --format=tar "${megatron_lm_commit}" \
    | tar -xf - -C "${temporary}/external/Megatron-Bridge/3rdparty/Megatron-LM"
  ln -s "${repo_root}/outputs" "${temporary}/outputs"

  jq -n \
    --arg source_commit "${source_commit}" \
    --arg source_tree "$(git rev-parse "${source_commit}^{tree}")" \
    --arg bridge_commit "${bridge_commit}" \
    --arg bridge_tree "$(git -C "${repo_root}/external/Megatron-Bridge" rev-parse "${bridge_commit}^{tree}")" \
    --arg megatron_lm_commit "${megatron_lm_commit}" \
    --arg megatron_lm_tree "$(git -C "${repo_root}/external/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse "${megatron_lm_commit}^{tree}")" \
    --arg created_at "$(date --iso-8601=seconds)" \
    '{schema: "stage1_l26_e32_submission_source_snapshot/v1",
      source_commit: $source_commit,
      source_tree: $source_tree,
      bridge_commit: $bridge_commit,
      bridge_tree: $bridge_tree,
      megatron_lm_commit: $megatron_lm_commit,
      megatron_lm_tree: $megatron_lm_tree,
      created_at: $created_at,
      execution_policy: "immutable_submission_commit_snapshot",
      mutable_checkout_after_submission_is_not_executed: true}' \
    > "${temporary}/submission_source_manifest.json"

  find "${temporary}" -type f -exec chmod a-w {} +
  mv "${temporary}" "${snapshot_root}"
  verify_submission_snapshot "${snapshot_root}" "${source_commit}"
  SOURCE_SNAPSHOT_ROOT="${snapshot_root}"
}

verify_source_lock() {
  command -v git >/dev/null || die "git is unavailable"
  command -v jq >/dev/null || die "jq is unavailable"
  command -v sha256sum >/dev/null || die "sha256sum is unavailable"
  [[ -x "${runtime_python}" ]] || die "missing runtime Python: ${runtime_python}"
  [[ -f "${launch_contract}" ]] || die "missing launch contract: ${launch_contract}"

  local actual_launch_sha
  actual_launch_sha="$(sha256_file "${launch_contract}")"
  [[ "${actual_launch_sha}" == "${expected_launch_contract_sha256}" ]] \
    || die "launch contract hash mismatch"
  jq -e '.schema == "stage1_l26_e32_cooldown_launch/v1"' "${launch_contract}" >/dev/null \
    || die "unexpected launch contract schema"

  local relative expected actual
  while IFS=$'\t' read -r relative expected; do
    [[ -f "${repo_root}/${relative}" ]] || die "missing locked file: ${relative}"
    actual="$(sha256_file "${repo_root}/${relative}")"
    [[ "${actual}" == "${expected}" ]] || die "source-lock mismatch: ${relative}"
  done < <(jq -r '.source_lock.files[] | [.path, .sha256] | @tsv' "${launch_contract}")

  expected="$(jq -r '.source_lock.pipeline_python_tree_sha256' "${launch_contract}")"
  actual="$(tracked_tree_sha256 src/multilingual_megatron/pipeline)"
  [[ "${actual}" == "${expected}" ]] || die "pipeline source tree hash mismatch"

  expected="$(jq -r '.source_lock.pipeline_config_tree_sha256' "${launch_contract}")"
  actual="$(tracked_tree_sha256 configs/pipeline)"
  [[ "${actual}" == "${expected}" ]] || die "pipeline config tree hash mismatch"

  expected="$(jq -r '.source_lock.bridge_commit' "${launch_contract}")"
  actual="$(git -C external/Megatron-Bridge rev-parse HEAD)"
  [[ "${actual}" == "${expected}" ]] || die "Megatron Bridge commit mismatch"
  [[ -z "$(git -C external/Megatron-Bridge status --porcelain)" ]] \
    || die "Megatron Bridge is not clean"

  expected="$(jq -r '.source_lock.megatron_lm_commit' "${launch_contract}")"
  actual="$(git -C external/Megatron-Bridge/3rdparty/Megatron-LM rev-parse HEAD)"
  [[ "${actual}" == "${expected}" ]] || die "Megatron-LM commit mismatch"
  [[ -z "$(git -C external/Megatron-Bridge/3rdparty/Megatron-LM status --porcelain)" ]] \
    || die "Megatron-LM is not clean"
}

committed_source_is_clean() {
  local paths=()
  mapfile -t paths < <(jq -r '.source_lock.committed_paths[]' "${launch_contract}")
  git ls-files --error-unmatch -- "${paths[@]}" >/dev/null 2>&1 \
    && git diff --quiet -- "${paths[@]}" \
    && git diff --cached --quiet -- "${paths[@]}"
}

verify_fresh_submission_namespace() {
  local run_root="${cooldown_root}/runs/stage1_l26_e32_k200_jointmoe_lr_cooldown"
  local marker
  for marker in .training_complete segment_results.json segment_results.partial.json training_run_manifest.json; do
    [[ ! -e "${run_root}/${marker}" ]] || die "cooldown output is not fresh: ${run_root}/${marker}"
  done
  if [[ -d "${run_root}" ]] && find "${run_root}" -path '*/checkpoints/*' -type f -print -quit | grep -q .; then
    die "cooldown output already contains checkpoint files"
  fi
}

verify_completed_trunk() {
  local trunk_job_id expected_state expected_exit accounting accounting_exit
  trunk_job_id="$(jq -r '.trunk.job_id' "${launch_contract}")"
  expected_state="$(jq -r '.trunk.required_terminal_state' "${launch_contract}")"
  expected_exit="$(jq -r '.trunk.required_exit_code' "${launch_contract}")"
  [[ -x "${slurm_bin}/sacct" ]] || die "sacct is unavailable: ${slurm_bin}/sacct"

  set +e
  accounting="$("${slurm_bin}/sacct" -X -j "${trunk_job_id}" \
    --format=JobIDRaw,State,ExitCode -n -P 2>&1)"
  accounting_exit=$?
  set -e
  (( accounting_exit == 0 )) || die "unable to verify terminal trunk ${trunk_job_id}: ${accounting}"
  TRUNK_TERMINAL="$(awk -F '|' -v job="${trunk_job_id}" \
    '$1 == job {print $2 "|" $3; exit}' <<<"${accounting}")"
  [[ "${TRUNK_TERMINAL}" == "${expected_state}|${expected_exit}" ]] \
    || die "trunk ${trunk_job_id} is not ${expected_state}|${expected_exit}: ${TRUNK_TERMINAL:-missing}"
}

build_sbatch_command() {
  SBATCH_COMMAND=(
    "${slurm_bin}/sbatch" --parsable
    --partition="$(jq -r '.scheduler.partition' "${launch_contract}")"
    --account="$(jq -r '.scheduler.account' "${launch_contract}")"
    --nodes="$(jq -r '.scheduler.nodes' "${launch_contract}")"
    --ntasks-per-node="$(jq -r '.scheduler.tasks_per_node' "${launch_contract}")"
    --cpus-per-task="$(jq -r '.scheduler.cpus_per_task' "${launch_contract}")"
    --gres="gpu:h100:$(jq -r '.scheduler.h100_gpus_per_node' "${launch_contract}")"
    --mem="$(jq -r '.scheduler.memory_per_node' "${launch_contract}")"
    --time="$(jq -r '.scheduler.time_limit' "${launch_contract}")"
    --signal=TERM@1800
    --export=WANDB_API_KEY
    --job-name=stage1_l26_e32_jointmoe_lr_cooldown_lX
    --chdir="${repo_root}"
    --output="${slurm_log_root}/cooldown-%j.slurm.log"
    "${script_path}" run
  )
}

print_plan() {
  build_sbatch_command
  printf 'Cooldown: verify trunk 2758779 COMPLETED|0:0 -> step 217233 parent -> final step 269634.\n'
  printf 'Resources: 2 nodes, 8 H100, 48:00:00; no reservation or nodelist.\n'
  printf 'Scheduler gate: direct submission only after a fresh sacct terminal-state check.\n'
  printf 'Command:\n'
  printf ' %q' "${SBATCH_COMMAND[@]}"
  printf '\n'
  if committed_source_is_clean; then
    echo "Locked launch paths are committed and clean."
  else
    echo "Preparation is valid, but submit remains fail-closed until the locked launch paths are committed and clean."
  fi
  if [[ -d "${submission_guard}" ]]; then
    echo "Submission is already claimed by ${submission_guard}; another submit is blocked."
  fi
}

submit_job() {
  committed_source_is_clean || die "locked launch paths must be tracked, committed, and clean"
  [[ -n "${WANDB_API_KEY:-}" ]] || die "WANDB_API_KEY is required but is never written to the script or receipts"
  [[ -x "${slurm_bin}/sbatch" ]] || die "sbatch is unavailable: ${slurm_bin}/sbatch"
  verify_completed_trunk

  local source_commit source_snapshot_manifest_sha
  source_commit="$(git rev-parse HEAD)"
  materialize_submission_snapshot "${source_commit}"
  source_snapshot_manifest_sha="$(sha256_file "${SOURCE_SNAPSHOT_ROOT}/submission_source_manifest.json")"

  mkdir -p "${slurm_log_root}" "${receipt_root}"
  build_sbatch_command

  if ! mkdir "${submission_guard}"; then
    die "cooldown submission is already claimed: ${submission_guard}"
  fi
  chmod 700 "${submission_guard}"

  local intent intent_tmp
  intent="${submission_guard}/intent.json"
  intent_tmp="${intent}.tmp"
  jq -n \
    --arg claimed_at "$(date --iso-8601=seconds)" \
    --arg source_commit "${source_commit}" \
    --arg source_snapshot_root "${SOURCE_SNAPSHOT_ROOT}" \
    --arg source_snapshot_manifest_sha256 "${source_snapshot_manifest_sha}" \
    --arg trunk_terminal "${TRUNK_TERMINAL}" \
    --arg launch_contract_sha256 "${expected_launch_contract_sha256}" \
    '{schema: "stage1_l26_e32_cooldown_submission_intent/v2",
      status: "submitting",
      claimed_at: $claimed_at,
      source_commit: $source_commit,
      execution_source_policy: "immutable_submission_commit_snapshot",
      source_snapshot_root: $source_snapshot_root,
      source_snapshot_manifest_sha256: $source_snapshot_manifest_sha256,
      scheduler_dependency: null,
      parent_gate: {mode: "terminal_parent_verified_before_submission", observed: $trunk_terminal},
      launch_contract_sha256: $launch_contract_sha256}' > "${intent_tmp}"
  chmod 600 "${intent_tmp}"
  mv "${intent_tmp}" "${intent}"

  local sbatch_response sbatch_exit sbatch_stderr failure_receipt failure_tmp
  sbatch_stderr="${submission_guard}/sbatch.stderr"
  set +e
  sbatch_response="$("${SBATCH_COMMAND[@]}" 2>"${sbatch_stderr}")"
  sbatch_exit=$?
  set -e
  if (( sbatch_exit != 0 )); then
    failure_receipt="${receipt_root}/cooldown_sbatch_failure.json"
    failure_tmp="${failure_receipt}.tmp"
    jq -n \
      --arg failed_at "$(date --iso-8601=seconds)" \
      --argjson exit_code "${sbatch_exit}" \
      --arg stderr "$(<"${sbatch_stderr}")" \
      '{schema: "stage1_l26_e32_cooldown_sbatch_failure/v1",
        status: "confirmed_not_submitted",
        failed_at: $failed_at,
        exit_code: $exit_code,
        stderr: $stderr,
        credentials_recorded: false}' > "${failure_tmp}"
    mv "${failure_tmp}" "${failure_receipt}"
    rm -f "${intent}" "${sbatch_stderr}"
    rmdir "${submission_guard}"
    die "sbatch failed; the one-shot guard was released after confirmed non-submission"
  fi

  local response_path response_tmp job_id job_id_path job_id_tmp
  response_path="${submission_guard}/sbatch_response.txt"
  response_tmp="${response_path}.tmp"
  printf '%s\n' "${sbatch_response}" > "${response_tmp}"
  chmod 600 "${response_tmp}"
  mv "${response_tmp}" "${response_path}"
  job_id="${sbatch_response}"
  job_id="${job_id%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || die "unexpected sbatch response: ${job_id}"
  job_id_path="${submission_guard}/job_id"
  job_id_tmp="${job_id_path}.tmp"
  printf '%s\n' "${job_id}" > "${job_id_tmp}"
  chmod 600 "${job_id_tmp}"
  mv "${job_id_tmp}" "${job_id_path}"

  local scheduler_snapshot scheduler_exit snapshot_path snapshot_tmp
  set +e
  scheduler_snapshot="$("${slurm_bin}/scontrol" show job "${job_id}" -o 2>&1)"
  scheduler_exit=$?
  set -e
  snapshot_path="${submission_guard}/scontrol_job.txt"
  snapshot_tmp="${snapshot_path}.tmp"
  printf '%s\n' "${scheduler_snapshot}" > "${snapshot_tmp}"
  chmod 600 "${snapshot_tmp}"
  mv "${snapshot_tmp}" "${snapshot_path}"

  local submitted_at script_sha script_mode receipt temporary
  local observed_state observed_reason observed_dependency observed_nodes observed_cpus observed_tres scheduler_verified
  local dependency_valid node_count_valid
  submitted_at="$(date --iso-8601=seconds)"
  script_sha="$(sha256_file "${script_path}")"
  script_mode="$(stat -c '%a' "${script_path}")"
  observed_state="$(sed -n 's/.* JobState=\([^ ]*\).*/\1/p' <<<"${scheduler_snapshot}")"
  observed_reason="$(sed -n 's/.* Reason=\([^ ]*\).*/\1/p' <<<"${scheduler_snapshot}")"
  observed_dependency="$(sed -n 's/.* Dependency=\([^ ]*\).*/\1/p' <<<"${scheduler_snapshot}")"
  observed_nodes="$(sed -n 's/.* NumNodes=\([^ ]*\).*/\1/p' <<<"${scheduler_snapshot}")"
  observed_cpus="$(sed -n 's/.* NumCPUs=\([^ ]*\).*/\1/p' <<<"${scheduler_snapshot}")"
  observed_tres="$(sed -n 's/.* TresPerNode=\([^ ]*\).*/\1/p' <<<"${scheduler_snapshot}")"
  dependency_valid=false
  if [[ -z "${observed_dependency}" || "${observed_dependency}" == "(null)" || "${observed_dependency}" == "null" ]] \
    && [[ "${TRUNK_TERMINAL}" == "COMPLETED|0:0" ]]; then
    dependency_valid=true
  fi
  node_count_valid=false
  if [[ "${observed_nodes}" == "2" || "${observed_nodes}" == "2-2" ]]; then
    node_count_valid=true
  fi
  scheduler_verified=false
  if (( scheduler_exit == 0 )) \
    && [[ "${dependency_valid}" == "true" ]] \
    && [[ "${node_count_valid}" == "true" ]] \
    && [[ "${observed_tres}" == *"gres/gpu:h100:4"* ]]; then
    scheduler_verified=true
  fi
  receipt="${receipt_root}/cooldown_submission_receipt_job${job_id}.json"
  temporary="${receipt}.tmp"
  jq -n \
    --arg submitted_at "${submitted_at}" \
    --argjson job_id "${job_id}" \
    --arg source_commit "${source_commit}" \
    --arg source_snapshot_root "${SOURCE_SNAPSHOT_ROOT}" \
    --arg source_snapshot_manifest_sha256 "${source_snapshot_manifest_sha}" \
    --arg launch_contract_sha256 "${expected_launch_contract_sha256}" \
    --arg launcher_sha256 "${script_sha}" \
    --arg launcher_mode "${script_mode}" \
    --arg observed_state "${observed_state}" \
    --arg observed_reason "${observed_reason}" \
    --arg observed_dependency "${observed_dependency}" \
    --arg observed_nodes "${observed_nodes}" \
    --arg observed_cpus "${observed_cpus}" \
    --arg observed_tres_per_node "${observed_tres}" \
    --arg trunk_terminal "${TRUNK_TERMINAL}" \
    --argjson dependency_valid "${dependency_valid}" \
    --argjson node_count_valid "${node_count_valid}" \
    --argjson scheduler_verified "${scheduler_verified}" \
    '{schema: "stage1_l26_e32_cooldown_submission_receipt/v1",
      status: (if $scheduler_verified then "submitted_terminal_parent_gated" else "submitted_scheduler_verification_failed" end),
      submitted_at: $submitted_at,
      job_id: $job_id,
      scheduler_dependency: null,
      parent_gate: {mode: "terminal_parent_verified_before_submission", observed: $trunk_terminal},
      source_commit: $source_commit,
      execution_source_policy: "immutable_submission_commit_snapshot",
      source_snapshot_root: $source_snapshot_root,
      source_snapshot_manifest_sha256: $source_snapshot_manifest_sha256,
      launch_contract_sha256: $launch_contract_sha256,
      launcher_sha256: $launcher_sha256,
      launcher_mode: $launcher_mode,
      requested_resources: {nodes: 2, h100_gpu_count: 8, time_limit: "48:00:00"},
      scheduler_observation: {
        valid: $scheduler_verified,
        state: $observed_state,
        reason: $observed_reason,
        dependency: $observed_dependency,
        dependency_valid: $dependency_valid,
        trunk_terminal: $trunk_terminal,
        nodes: $observed_nodes,
        node_count_valid: $node_count_valid,
        cpus: $observed_cpus,
        tres_per_node: $observed_tres_per_node},
      credentials_recorded: false}' > "${temporary}"
  mv "${temporary}" "${receipt}"
  [[ "${scheduler_verified}" == "true" ]] \
    || die "job ${job_id} was submitted but scheduler verification failed; duplicate submission remains locked"
  printf 'Submitted cooldown job %s after verifying trunk %s\nReceipt: %s\n' "${job_id}" "${TRUNK_TERMINAL}" "${receipt}"
}

run_job() {
  [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] \
    || die "run is only valid inside the submitted Slurm allocation"
  [[ -n "${WANDB_API_KEY:-}" ]] || die "WANDB_API_KEY was not transported into the allocation"
  export PATH="/usr/local/bin:/usr/bin:/bin"

  local intent source_commit execution_root snapshot_manifest_sha expected_snapshot_manifest_sha parent_terminal
  intent="${submission_guard}/intent.json"
  [[ -f "${intent}" ]] || die "submission intent is missing: ${intent}"
  source_commit="$(jq -r '.source_commit' "${intent}")"
  execution_root="$(jq -r '.source_snapshot_root' "${intent}")"
  expected_snapshot_manifest_sha="$(jq -r '.source_snapshot_manifest_sha256' "${intent}")"
  parent_terminal="$(jq -r '.parent_gate.observed' "${intent}")"
  [[ "${parent_terminal}" == "COMPLETED|0:0" ]] \
    || die "submission intent lacks the successful terminal-parent proof"
  [[ "${execution_root}" == "${source_snapshot_store}/${source_commit}" ]] \
    || die "submission snapshot path is not bound to the submission commit"
  verify_submission_snapshot "${execution_root}" "${source_commit}"
  snapshot_manifest_sha="$(sha256_file "${execution_root}/submission_source_manifest.json")"
  [[ "${snapshot_manifest_sha}" == "${expected_snapshot_manifest_sha}" ]] \
    || die "submission snapshot manifest hash mismatch"

  export SLURM_EXPORT_ENV=ALL
  [[ -r "${module_init}" ]] || die "unreadable Lmod init: ${module_init}"
  set +u
  source "${module_init}"
  set -u
  module purge
  module load toolchain/foss/2024a system/CUDA/12.6.0 lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
  set +u
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "$(dirname "$(dirname "${runtime_python}")")"
  set -u
  export PYTHONPATH="${execution_root}/src:${execution_root}:${execution_root}/external/Megatron-Bridge/src:${execution_root}/external/Megatron-Bridge/3rdparty/Megatron-LM"
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1

  local nodes_json start_receipt temporary checkout_commit_at_start
  nodes_json="$("${slurm_bin}/scontrol" show hostnames "${SLURM_JOB_NODELIST}" | jq -Rsc 'split("\n") | map(select(length > 0))')"
  [[ "$(jq 'length' <<<"${nodes_json}")" == "2" ]] || die "expected exactly two allocated nodes"
  checkout_commit_at_start="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf 'unavailable')"
  mkdir -p "${receipt_root}"
  start_receipt="${receipt_root}/cooldown_start_receipt_job${SLURM_JOB_ID}.json"
  temporary="${start_receipt}.tmp"
  jq -n \
    --arg started_at "$(date --iso-8601=seconds)" \
    --argjson job_id "${SLURM_JOB_ID}" \
    --argjson nodes "${nodes_json}" \
    --arg source_commit "${source_commit}" \
    --arg checkout_commit_at_start "${checkout_commit_at_start}" \
    --arg execution_root "${execution_root}" \
    --arg source_snapshot_manifest_sha256 "${snapshot_manifest_sha}" \
    --arg parent_terminal "${parent_terminal}" \
    --arg launch_contract_sha256 "${expected_launch_contract_sha256}" \
    '{schema: "stage1_l26_e32_cooldown_start_receipt/v2",
      status: "submission_commit_snapshot_verified_starting",
      started_at: $started_at,
      job_id: $job_id,
      allocated_nodes: $nodes,
      chosen_node_subset: $nodes,
      h100_gpu_count: 8,
      scheduler_dependency: null,
      parent_gate: {mode: "terminal_parent_verified_before_submission", observed: $parent_terminal},
      trunk_job_id: 2758779,
      source_commit: $source_commit,
      checkout_commit_at_start: $checkout_commit_at_start,
      execution_root: $execution_root,
      execution_source_policy: "immutable_submission_commit_snapshot",
      source_snapshot_manifest_sha256: $source_snapshot_manifest_sha256,
      launch_contract_sha256: $launch_contract_sha256,
      locked_surface_valid: true,
      credentials_recorded: false}' > "${temporary}"
  mv "${temporary}" "${start_receipt}"

  cd "${execution_root}"
  exec "${runtime_python}" -m multilingual_megatron.pipeline \
    --config configs/pipeline \
    --profile stage1_l26_e32_k200_jointmoe_lr_cooldown_production_v1 \
    --start-at train \
    --stop-after train \
    --force-stage train \
    --executor-type local \
    --attempt-id "${attempt_id}"
}

action="${1:-plan}"
case "${action}" in
  plan)
    verify_source_lock
    verify_fresh_submission_namespace
    print_plan
    ;;
  submit)
    verify_source_lock
    verify_fresh_submission_namespace
    print_plan
    submit_job
    ;;
  run)
    run_job
    ;;
  self-test-spooled-root)
    printf '%s\n' "${repo_root}"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
