#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="/opt/conda/envs/bridge-train-a100"
SYSROOT_LIB="${ENV_ROOT}/x86_64-conda-linux-gnu/sysroot/lib64"
PYTHON_BIN="${ENV_ROOT}/bin/python"
LOADER="${SYSROOT_LIB}/ld-linux-x86-64.so.2"

LIBRARY_PATH="${SYSROOT_LIB}:${ENV_ROOT}/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LIBRARY_PATH="${LIBRARY_PATH}:${LD_LIBRARY_PATH}"
fi

exec "${LOADER}" --library-path "${LIBRARY_PATH}" "${PYTHON_BIN}" "$@"
