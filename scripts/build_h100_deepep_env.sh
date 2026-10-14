#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${1:-${H100_DEEPEP_ENV_PREFIX:-/opt/conda/envs/bridge-train-h100-deepep-cu126}}"
BASE_ENV="${H100_DEEPEP_BASE_ENV:-bridge-train}"
CONDA_BASE="${CONDA_BASE:-/opt/software/cluster/EB-SW/software/Miniforge3/25.3.0-3}"
BUILD_ROOT="${H100_DEEPEP_BUILD_ROOT:-/scratch/shared_cache/tmp/h100_deepep_sm90_build}"
DEEPEP_SOURCE="${H100_DEEPEP_SOURCE:-/scratch/shared_cache/tmp/DeepEP}"
DEEPEP_REF="${H100_DEEPEP_REF:-v1.2.1}"
DEEPEP_SAFE_REF="${DEEPEP_REF//[^A-Za-z0-9_.-]/_}"
DEEPEP_BUILD_SRC="${H100_DEEPEP_BUILD_SRC:-${BUILD_ROOT}/DeepEP-${DEEPEP_SAFE_REF}-sm90}"
EXPECT_HYBRIDEP="${H100_DEEPEP_EXPECT_HYBRIDEP:-0}"
RDMA_CORE_MODULE="${H100_DEEPEP_RDMA_CORE_MODULE:-rdma-core-60.1-gcc-14.3.0-jwu7afel}"
RDMA_CORE_ROOT="${H100_DEEPEP_RDMA_CORE_ROOT:-}"
NVTX_LIB_DIR="${H100_DEEPEP_NVTX_LIB_DIR:-}"
HYBRIDEP_NVTX_LIBRARY="${H100_DEEPEP_HYBRIDEP_NVTX_LIBRARY:-nvToolsExt}"
JOBS="${H100_DEEPEP_BUILD_JOBS:-16}"
export DEEPEP_BUILD_SRC
export EXPECT_HYBRIDEP
export HYBRIDEP_NVTX_LIBRARY

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if type module >/dev/null 2>&1; then
  module purge
  module load toolchain/foss/2024a
  module load system/CUDA/12.6.0
  module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
  module load .spack >/dev/null 2>&1 || true
  module load "${RDMA_CORE_MODULE}" >/dev/null 2>&1 || true
fi

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi
if [[ -z "${RDMA_CORE_ROOT}" ]] && command -v ibv_devinfo >/dev/null 2>&1; then
  RDMA_CORE_ROOT="$(cd "$(dirname "$(command -v ibv_devinfo)")/.." && pwd)"
fi
if [[ -z "${NVTX_LIB_DIR}" && -d "${CUDA_HOME:-/opt/software/cluster/EB-SW/software/CUDA/12.6.0}/targets/x86_64-linux/lib" ]]; then
  NVTX_LIB_DIR="${CUDA_HOME:-/opt/software/cluster/EB-SW/software/CUDA/12.6.0}/targets/x86_64-linux/lib"
fi
export CUDA_HOME="${CUDA_HOME:-/opt/software/cluster/EB-SW/software/CUDA/12.6.0}"
export RDMA_CORE_ROOT
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
export MAX_JOBS="${MAX_JOBS:-${JOBS}}"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

mkdir -p "${BUILD_ROOT}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "[INFO] Creating isolated env ${ENV_PREFIX} from ${BASE_ENV}"
  conda create --prefix "${ENV_PREFIX}" --clone "${BASE_ENV}" -y
fi

conda activate "${ENV_PREFIX}"
PYTHON="${ENV_PREFIX}/bin/python"

if [[ ! -d "${DEEPEP_SOURCE}/.git" ]]; then
  git clone https://github.com/deepseek-ai/DeepEP.git "${DEEPEP_SOURCE}"
fi

if [[ ! -d "${DEEPEP_BUILD_SRC}/.git" ]]; then
  git clone "${DEEPEP_SOURCE}" "${DEEPEP_BUILD_SRC}"
fi
git -C "${DEEPEP_BUILD_SRC}" fetch --all --tags
DEEPEP_RESOLVED_REF="$(
  git -C "${DEEPEP_BUILD_SRC}" rev-parse --verify "${DEEPEP_REF}^{commit}" 2>/dev/null || true
)"
if [[ -z "${DEEPEP_RESOLVED_REF}" ]]; then
  SOURCE_REF="${DEEPEP_REF}"
  if [[ "${SOURCE_REF}" == origin/* ]]; then
    SOURCE_REF="refs/remotes/${SOURCE_REF}"
  fi
  if git -C "${DEEPEP_SOURCE}" rev-parse --verify "${SOURCE_REF}^{commit}" >/dev/null 2>&1; then
    git -C "${DEEPEP_BUILD_SRC}" fetch "${DEEPEP_SOURCE}" "${SOURCE_REF}"
    DEEPEP_RESOLVED_REF="$(
      git -C "${DEEPEP_BUILD_SRC}" rev-parse --verify "FETCH_HEAD^{commit}" 2>/dev/null || true
    )"
  fi
fi
if [[ -z "${DEEPEP_RESOLVED_REF}" ]]; then
  echo "Could not resolve DeepEP ref ${DEEPEP_REF}" >&2
  exit 1
fi
git -C "${DEEPEP_BUILD_SRC}" checkout --detach "${DEEPEP_RESOLVED_REF}"

"${PYTHON}" - <<'PY'
import os
from pathlib import Path

setup_py = Path(os.environ["DEEPEP_BUILD_SRC"]) / "setup.py"
text = setup_py.read_text(encoding="utf-8")
old = "'/usr/local/cuda/include/cccl'"
new = "os.path.join(os.environ.get('CUDA_HOME', '/usr/local/cuda'), 'include', 'cccl')"
if old in text:
    text = text.replace(old, new)
legacy_disable = "    disable_nvshmem = False\n"
patched_disable = """    disable_nvshmem = os.getenv('DEEPEP_DISABLE_NVSHMEM', '0').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }
"""
if legacy_disable in text and "DEEPEP_DISABLE_NVSHMEM" not in text:
    text = text.replace(legacy_disable, patched_disable, 1)
legacy_probe = "    if nvshmem_dir is None:\n"
patched_probe = """    if disable_nvshmem:
        nvshmem_dir = None
        print('Warning: `DEEPEP_DISABLE_NVSHMEM` is set. All internode and low-latency features are disabled\\n')
    elif nvshmem_dir is None:
"""
if legacy_probe in text and "elif nvshmem_dir is None" not in text:
    text = text.replace(legacy_probe, patched_probe, 1)
hybridep_nvtx_library = os.environ.get("HYBRIDEP_NVTX_LIBRARY", "").strip()
if hybridep_nvtx_library and '"cuda", "nvtx3interop"' in text:
    text = text.replace('"cuda", "nvtx3interop"', f'"cuda", "{hybridep_nvtx_library}"')
setup_py.write_text(text, encoding="utf-8")
PY

export PIP_NO_BUILD_ISOLATION=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${BUILD_ROOT}/pip-cache}"
export EP_JIT_CACHE_DIR="${EP_JIT_CACHE_DIR:-${BUILD_ROOT}/jit-cache}"
export TORCH_CUDA_ARCH_LIST="${H100_DEEPEP_CUDA_ARCH_LIST:-9.0}"
export DISABLE_SM90_FEATURES=0
export DISABLE_AGGRESSIVE_PTX_INSTRS="${DISABLE_AGGRESSIVE_PTX_INSTRS:-1}"
export CPATH="${CUDA_HOME}/include:${CUDA_HOME}/include/cccl:${CPATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-L${CUDA_HOME}/lib64/stubs ${LDFLAGS:-}"
if [[ -n "${RDMA_CORE_ROOT}" && -d "${RDMA_CORE_ROOT}/include" ]]; then
  export CPATH="${RDMA_CORE_ROOT}/include:${CPATH}"
fi
if [[ -n "${RDMA_CORE_ROOT}" && -d "${RDMA_CORE_ROOT}/lib64" ]]; then
  export LIBRARY_PATH="${RDMA_CORE_ROOT}/lib64:${LIBRARY_PATH}"
  export LD_LIBRARY_PATH="${RDMA_CORE_ROOT}/lib64:${LD_LIBRARY_PATH}"
fi
if [[ -n "${NVTX_LIB_DIR}" && -d "${NVTX_LIB_DIR}" ]]; then
  export LIBRARY_PATH="${NVTX_LIB_DIR}:${LIBRARY_PATH}"
  export LD_LIBRARY_PATH="${NVTX_LIB_DIR}:${LD_LIBRARY_PATH}"
  export LDFLAGS="-L${NVTX_LIB_DIR} ${LDFLAGS}"
fi

echo "[INFO] DeepEP ref=${DEEPEP_REF} build_src=${DEEPEP_BUILD_SRC}"
if [[ "${EXPECT_HYBRIDEP}" == "1" ]]; then
  echo "[INFO] HybridEP symbol will be required after install"
fi
echo "[INFO] CUDA_HOME=${CUDA_HOME}"
echo "[INFO] RDMA_CORE_ROOT=${RDMA_CORE_ROOT:-<not found>}"
echo "[INFO] NVTX_LIB_DIR=${NVTX_LIB_DIR:-<not found>}"
echo "[INFO] HYBRIDEP_NVTX_LIBRARY=${HYBRIDEP_NVTX_LIBRARY}"
echo "[INFO] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

"${PYTHON}" -m pip install --upgrade "setuptools<80" wheel
"${PYTHON}" -m pip install --no-build-isolation --no-cache-dir -v "${DEEPEP_BUILD_SRC}"

if [[ "${EXPECT_HYBRIDEP}" == "1" || "${H100_DEEPEP_PATCH_HYBRIDEP_CUDA126:-0}" == "1" ]]; then
  "${PYTHON}" "${PWD}/scripts/patch_h100_hybridep_cuda126.py"
fi

"${PYTHON}" - <<'PY'
import importlib.metadata as md
import os
import torch
from deep_ep import Buffer

try:
    from deep_ep import HybridEPBuffer  # noqa: F401
    has_hybridep = True
except Exception as exc:
    has_hybridep = False
    hybridep_error = f"{type(exc).__name__}: {exc}"

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("deep-ep", md.version("deep-ep"))
print("deep_ep_sm90_compiled", Buffer.is_sm90_compiled())
print("deep_ep_has_hybridep", has_hybridep)
if not has_hybridep:
    print("hybridep_import_error", hybridep_error)
if not Buffer.is_sm90_compiled():
    raise SystemExit("DeepEP build did not enable SM90 features")
if os.environ.get("EXPECT_HYBRIDEP", "").strip() == "1" and not has_hybridep:
    raise SystemExit("DeepEP build did not expose HybridEPBuffer")
PY

echo "[OK] H100 DeepEP env ready: ${ENV_PREFIX}"
echo "[NEXT] H100_DEEPEP_ENV_PREFIX=${ENV_PREFIX} bash scripts/submit_h100_deepep_preflight.sh"
