#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${BRIDGE_TRAIN_ENV_NAME:-bridge-train}"
CONDA_BASE="${CONDA_BASE:-/opt/software/cluster/EB-SW/software/Miniforge3/25.3.0-3}"

if [[ "${BRIDGE_TRAIN_LOAD_MODULES:-1}" == "1" ]]; then
  if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  if type module >/dev/null 2>&1; then
    module purge
    module load toolchain/foss/2024a
    module load system/CUDA/12.6.0
    module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
  fi
fi

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  export CUDA_HOME
fi
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi

if [[ "${BRIDGE_TRAIN_RECREATE:-0}" == "1" ]] && conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env remove -n "${ENV_NAME}" -y
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${REPO_ROOT}/envs/bridge-train.yml" --prune
else
  conda env create -f "${REPO_ROOT}/envs/bridge-train.yml"
fi

set +u
conda activate "${ENV_NAME}"
set -u

python -m pip install --upgrade pip "setuptools<80" wheel
PY_SITE="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
HOOK_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEHOOK_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${HOOK_DIR}" "${DEHOOK_DIR}"
cat >"${HOOK_DIR}/bridge-train-cuda-libs.sh" <<'SH'
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

_bridge_train_site="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
_bridge_train_include_dirs=()
_bridge_train_lib_dirs=()
for _bridge_train_include_dir in "${_bridge_train_site}"/nvidia/*/include; do
  [[ -d "${_bridge_train_include_dir}" ]] && _bridge_train_include_dirs+=("${_bridge_train_include_dir}")
done
for _bridge_train_lib_dir in "${_bridge_train_site}"/nvidia/*/lib; do
  [[ -d "${_bridge_train_lib_dir}" ]] && _bridge_train_lib_dirs+=("${_bridge_train_lib_dir}")
done

export _BRIDGE_TRAIN_OLD_CPATH="${CPATH:-}"
export _BRIDGE_TRAIN_OLD_LIBRARY_PATH="${LIBRARY_PATH:-}"
export _BRIDGE_TRAIN_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ "${#_bridge_train_include_dirs[@]}" -gt 0 ]]; then
  IFS=:
  export CPATH="${_bridge_train_include_dirs[*]}${CPATH:+:${CPATH}}"
  unset IFS
fi
if [[ "${#_bridge_train_lib_dirs[@]}" -gt 0 ]]; then
  IFS=:
  export LIBRARY_PATH="${_bridge_train_lib_dirs[*]}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${_bridge_train_lib_dirs[*]}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  unset IFS
fi
unset _bridge_train_site _bridge_train_include_dirs _bridge_train_lib_dirs
unset _bridge_train_include_dir _bridge_train_lib_dir
SH
cat >"${DEHOOK_DIR}/bridge-train-cuda-libs.sh" <<'SH'
if [[ -n "${_BRIDGE_TRAIN_OLD_CPATH+x}" ]]; then
  export CPATH="${_BRIDGE_TRAIN_OLD_CPATH}"
  unset _BRIDGE_TRAIN_OLD_CPATH
fi
if [[ -n "${_BRIDGE_TRAIN_OLD_LIBRARY_PATH+x}" ]]; then
  export LIBRARY_PATH="${_BRIDGE_TRAIN_OLD_LIBRARY_PATH}"
  unset _BRIDGE_TRAIN_OLD_LIBRARY_PATH
fi
if [[ -n "${_BRIDGE_TRAIN_OLD_LD_LIBRARY_PATH+x}" ]]; then
  export LD_LIBRARY_PATH="${_BRIDGE_TRAIN_OLD_LD_LIBRARY_PATH}"
  unset _BRIDGE_TRAIN_OLD_LD_LIBRARY_PATH
fi
SH
NVIDIA_INCLUDE_DIRS=()
NVIDIA_LIB_DIRS=()
for include_dir in "${PY_SITE}"/nvidia/*/include; do
  [[ -d "${include_dir}" ]] && NVIDIA_INCLUDE_DIRS+=("${include_dir}")
done
for lib_dir in "${PY_SITE}"/nvidia/*/lib; do
  [[ -d "${lib_dir}" ]] && NVIDIA_LIB_DIRS+=("${lib_dir}")
done
if [[ "${#NVIDIA_INCLUDE_DIRS[@]}" -gt 0 ]]; then
  IFS=:
  export CPATH="${NVIDIA_INCLUDE_DIRS[*]}${CPATH:+:${CPATH}}"
  unset IFS
fi
if [[ "${#NVIDIA_LIB_DIRS[@]}" -gt 0 ]]; then
  IFS=:
  export LIBRARY_PATH="${NVIDIA_LIB_DIRS[*]}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS[*]}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  unset IFS
fi
python -m pip install --no-build-isolation --no-cache-dir \
  "transformer-engine[pytorch]>=2.14,<2.15" \
  "causal-conv1d>=1.5,<1.6" \
  "mamba-ssm>=2.3,<2.4"

python -m pip install --no-deps --no-build-isolation -e "${REPO_ROOT}/external/Megatron-Bridge"

if [[ "${INSTALL_APEX:-0}" == "1" ]]; then
  python -m pip install --no-build-isolation --no-cache-dir \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    "git+https://github.com/NVIDIA/apex.git"
fi

python - <<'PY'
import importlib

required = [
    "torch",
    "transformer_engine.pytorch",
    "causal_conv1d",
    "mamba_ssm",
    "pybind11",
    "hydra",
    "tensorboard",
    "sentencepiece",
    "modelopt",
    "nvidia_resiliency_ext",
    "megatron.bridge",
]

missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

if missing:
    raise SystemExit("Bridge train env verification failed:\n" + "\n".join(missing))

print("bridge-train environment verified")
PY
