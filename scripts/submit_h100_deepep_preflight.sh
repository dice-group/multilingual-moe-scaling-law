#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${H100_DEEPEP_ENV_PREFIX:-/opt/conda/envs/bridge-train-h100-deepep-cu126}"
OUT_DIR="${H100_DEEPEP_PREFLIGHT_OUT:-${REPO_ROOT}/outputs/pipeline_runs/h100_moe_goal_loop_phase01_20260621/analysis/deepep_preflight}"
JOB_NAME="${H100_DEEPEP_PREFLIGHT_JOB_NAME:-h100_deepep_preflight}"
TIME_LIMIT="${H100_DEEPEP_PREFLIGHT_TIME:-00:05:00}"
GPUS="${H100_DEEPEP_PREFLIGHT_GPUS:-2}"
CPUS="${H100_DEEPEP_PREFLIGHT_CPUS:-16}"
MEM="${H100_DEEPEP_PREFLIGHT_MEM:-128G}"
REQUIRE_HYBRIDEP="${H100_DEEPEP_PREFLIGHT_REQUIRE_HYBRIDEP:-0}"
HYBRIDEP_JIT="${H100_DEEPEP_PREFLIGHT_HYBRIDEP_JIT:-0}"
HYBRIDEP_LOCAL_EXPERTS="${H100_DEEPEP_PREFLIGHT_LOCAL_EXPERTS:-16}"
HYBRIDEP_FUSED="${H100_DEEPEP_PREFLIGHT_HYBRIDEP_FUSED:-0}"
HYBRIDEP_NUM_SMS="${H100_DEEPEP_PREFLIGHT_HYBRIDEP_NUM_SMS:-16}"
RDMA_CORE_MODULE="${H100_DEEPEP_RDMA_CORE_MODULE:-rdma-core-60.1-gcc-14.3.0-jwu7afel}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "H100 DeepEP env not found: ${ENV_PREFIX}" >&2
  echo "Run scripts/build_h100_deepep_env.sh first." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

WRAP_CMD="module purge; \
module load toolchain/foss/2024a; \
module load system/CUDA/12.6.0; \
module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0; \
module load .spack >/dev/null 2>&1 || true; \
module load ${RDMA_CORE_MODULE} >/dev/null 2>&1 || true; \
if [ -z \"\${CUDA_HOME:-}\" ] && command -v nvcc >/dev/null 2>&1; then CUDA_HOME=\"\$(cd \"\$(dirname \"\$(command -v nvcc)\")/..\" && pwd)\"; fi; \
CUDA_HOME=\"\${CUDA_HOME:-/opt/software/cluster/EB-SW/software/CUDA/12.6.0}\"; \
RDMA_CORE_ROOT=\"\"; \
if command -v ibv_devinfo >/dev/null 2>&1; then RDMA_CORE_ROOT=\"\$(cd \"\$(dirname \"\$(command -v ibv_devinfo)\")/..\" && pwd)\"; fi; \
if [ -n \"\${RDMA_CORE_ROOT}\" ] && [ -d \"\${RDMA_CORE_ROOT}/lib64\" ]; then export LD_LIBRARY_PATH=\"\${RDMA_CORE_ROOT}/lib64:\${LD_LIBRARY_PATH:-}\"; fi; \
if [ -d \"\${CUDA_HOME}/targets/x86_64-linux/lib\" ]; then export LD_LIBRARY_PATH=\"\${CUDA_HOME}/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}\"; fi; \
export LD_LIBRARY_PATH=\"${ENV_PREFIX}/lib:${ENV_PREFIX}/lib/python3.12/site-packages/nvidia/nvshmem/lib:\${LD_LIBRARY_PATH:-}\"; \
cd ${REPO_ROOT}; \
export PYTHONPATH=src:.:external/Megatron-Bridge/src:external/Megatron-Bridge/3rdparty/Megatron-LM; \
export OMP_NUM_THREADS=1; \
export DEEPEP_PREFLIGHT_REQUIRE_HYBRIDEP=${REQUIRE_HYBRIDEP}; \
export DEEPEP_PREFLIGHT_HYBRIDEP_JIT=${HYBRIDEP_JIT}; \
export DEEPEP_PREFLIGHT_LOCAL_EXPERTS=${HYBRIDEP_LOCAL_EXPERTS}; \
export DEEPEP_PREFLIGHT_HYBRIDEP_FUSED=${HYBRIDEP_FUSED}; \
export DEEPEP_PREFLIGHT_HYBRIDEP_NUM_SMS=${HYBRIDEP_NUM_SMS}; \
${ENV_PREFIX}/bin/torchrun --standalone --nproc_per_node=${GPUS} scripts/h100_deepep_preflight.py"

sbatch \
  --partition=gpu_h100 \
  --gres="gpu:h100:${GPUS}" \
  --ntasks=1 \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME_LIMIT}" \
  --job-name="${JOB_NAME}" \
  --output="${OUT_DIR}/${JOB_NAME}.%j.out" \
  --error="${OUT_DIR}/${JOB_NAME}.%j.err" \
  --wrap="${WRAP_CMD}"
