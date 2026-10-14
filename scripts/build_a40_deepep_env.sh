#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${1:-/opt/conda/envs/bridge-train-a40-deepep-cu126}"
BUILD_ROOT="${A40_DEEPEP_BUILD_ROOT:-/scratch/shared_cache/tmp/a40_deepep_nvshmem_build}"
NVSHMEM_VERSION="${NVSHMEM_VERSION:-v3.5.19-1}"
DEEPEP_BRANCH="${DEEPEP_BRANCH:-nvshmem_deepep_gcp}"
DEEPEP_INSTALL_MODE="${A40_DEEPEP_INSTALL_MODE:-sm80_intranode}"
JOBS="${A40_DEEPEP_BUILD_JOBS:-16}"

module purge
module load system/CUDA/12.6.0
module load mpi/OpenMPI/5.0.3-GCC-13.3.0
module load lib/GDRCopy/2.4.1-GCCcore-13.3.0

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

export CUDA_HOME="${CUDA_HOME:-/opt/software/cluster/EB-SW/software/CUDA/12.6.0}"
export MPI_HOME="$(dirname "$(dirname "$(which mpicc)")")"
export GDRCOPY_HOME="${EBROOTGDRCOPY:?GDRCopy module did not set EBROOTGDRCOPY}"
export NVSHMEM_SRC="${BUILD_ROOT}/nvshmem"
export NVSHMEM_DIR="${ENV_PREFIX}/nvshmem-3.5.19-1/install"
export DEEPEP_SRC="${BUILD_ROOT}/DeepEP"
export PIP_CACHE_DIR="${BUILD_ROOT}/pip-cache"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_HOME}/lib64:${GDRCOPY_HOME}/lib:/usr/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${NVSHMEM_DIR}/lib:${CUDA_HOME}/lib64:${GDRCOPY_HOME}/lib:/usr/lib64:${LIBRARY_PATH:-}"
export CPATH="/usr/include:${CUDA_HOME}/include:${GDRCOPY_HOME}/include:${CPATH:-}"
if [[ -d "${CUDA_HOME}/include/cccl" ]]; then
  export CPATH="${CUDA_HOME}/include/cccl:${CPATH}"
fi

mkdir -p "${BUILD_ROOT}" "${NVSHMEM_DIR}" "${PIP_CACHE_DIR}"

if [[ ! -d "${NVSHMEM_SRC}/.git" ]]; then
  git clone --branch "${NVSHMEM_VERSION}" https://github.com/NVIDIA/nvshmem.git "${NVSHMEM_SRC}"
else
  git -C "${NVSHMEM_SRC}" fetch --tags origin
  git -C "${NVSHMEM_SRC}" checkout "${NVSHMEM_VERSION}"
fi

cmake -S "${NVSHMEM_SRC}" -B "${NVSHMEM_SRC}/build-a40-sm80" \
  -DCMAKE_INSTALL_PREFIX="${NVSHMEM_DIR}" \
  -DCUDA_HOME="${CUDA_HOME}" \
  -DMPI_HOME="${MPI_HOME}" \
  -DMPI_C_COMPILER="$(which mpicc)" \
  -DMPI_CXX_COMPILER="$(which mpicxx)" \
  -DNVSHMEM_MPI_SUPPORT=OFF \
  -DNVSHMEM_IBRC_SUPPORT=ON \
  -DNVSHMEM_IBGDA_SUPPORT=ON \
  -DNVSHMEM_IBDEVX_SUPPORT=OFF \
  -DNVSHMEM_UCX_SUPPORT=OFF \
  -DNVSHMEM_SHMEM_SUPPORT=OFF \
  -DNVSHMEM_PMIX_SUPPORT=OFF \
  -DNVSHMEM_USE_NCCL=OFF \
  -DNVSHMEM_USE_GDRCOPY=ON \
  -DGDRCOPY_HOME="${GDRCOPY_HOME}" \
  -DNVSHMEM_USE_MLX5DV=ON \
  -DNVSHMEM_BUILD_TESTS=OFF \
  -DNVSHMEM_BUILD_EXAMPLES=OFF \
  -DNVSHMEM_BUILD_PYTHON_LIB=OFF \
  -DNVSHMEM_BUILD_BITCODE_LIBRARY=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build "${NVSHMEM_SRC}/build-a40-sm80" -j "${JOBS}"
cmake --install "${NVSHMEM_SRC}/build-a40-sm80"

if [[ ! -d "${DEEPEP_SRC}/.git" ]]; then
  git clone https://github.com/zhongbozhu/DeepEP.git "${DEEPEP_SRC}"
else
  git -C "${DEEPEP_SRC}" fetch origin
fi
git -C "${DEEPEP_SRC}" checkout "${DEEPEP_BRANCH}"

export PIP_NO_BUILD_ISOLATION=1
python -m pip uninstall -y deep_ep deepep || true

if [[ "${DEEPEP_INSTALL_MODE}" == "sm80_intranode" ]]; then
  # DeepEP V1's NVSHMEM internode kernels use SM90-only PTX instructions
  # (TMA/mbarrier/elect). On A40/SM80 the supported build is intranode-only.
  python - <<'PY'
import os
from pathlib import Path

setup_py = Path(os.environ["DEEPEP_SRC"]) / "setup.py"
text = setup_py.read_text(encoding="utf-8")
old = "    disable_nvshmem = False\n    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)\n"
new = (
    "    disable_nvshmem = bool(int(os.getenv('DEEPEP_FORCE_DISABLE_NVSHMEM', '0')))\n"
    "    nvshmem_dir = None if disable_nvshmem else os.getenv('NVSHMEM_DIR', None)\n"
)
if old in text:
    text = text.replace(old, new)
old = "    if nvshmem_dir is None:\n        try:\n"
new = "    if disable_nvshmem:\n        pass\n    elif nvshmem_dir is None:\n        try:\n"
if old in text:
    text = text.replace(old, new)
setup_py.write_text(text, encoding="utf-8")

deep_ep_cpp = Path(os.environ["DEEPEP_SRC"]) / "csrc" / "deep_ep.cpp"
text = deep_ep_cpp.read_text(encoding="utf-8")
old = """void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < num_ranks);
    internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, at::cuda::getCurrentCUDAStream());
}

void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);

    internode_ll::query_mask_buffer(
        mask_buffer_ptr, num_ranks, reinterpret_cast<int*>(mask_status.data_ptr()), at::cuda::getCurrentCUDAStream());
}

void Buffer::low_latency_clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, at::cuda::getCurrentCUDAStream());
}
"""
new = """void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < num_ranks);
    internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, at::cuda::getCurrentCUDAStream());
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);

    internode_ll::query_mask_buffer(
        mask_buffer_ptr, num_ranks, reinterpret_cast<int*>(mask_status.data_ptr()), at::cuda::getCurrentCUDAStream());
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

void Buffer::low_latency_clean_mask_buffer() {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, at::cuda::getCurrentCUDAStream());
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}
"""
if old in text:
    text = text.replace(old, new)
deep_ep_cpp.write_text(text, encoding="utf-8")
PY
  rm -rf "${DEEPEP_SRC}/build" "${DEEPEP_SRC}/deep_ep.egg-info"
  DISABLE_SM90_FEATURES=1 \
    DEEPEP_FORCE_DISABLE_NVSHMEM=1 \
    TORCH_CUDA_ARCH_LIST="8.0" \
    python -m pip install --no-build-isolation -v "${DEEPEP_SRC}"
elif [[ "${DEEPEP_INSTALL_MODE}" == "sm80_nvshmem" ]]; then
  TORCH_CUDA_ARCH_LIST="8.0" python -m pip install --no-build-isolation -v "${DEEPEP_SRC}"
else
  echo "Unknown A40_DEEPEP_INSTALL_MODE=${DEEPEP_INSTALL_MODE}" >&2
  exit 2
fi

python - <<'PY'
import importlib.util
import pathlib
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("deepep", bool(importlib.util.find_spec("deepep")))
print("deep_ep", bool(importlib.util.find_spec("deep_ep")))
if importlib.util.find_spec("deep_ep"):
    from deep_ep import Buffer
    print("deep_ep_sm90_compiled", Buffer.is_sm90_compiled())
print("nvshmem_install", pathlib.Path(__import__("os").environ["NVSHMEM_DIR"]).exists())
PY
