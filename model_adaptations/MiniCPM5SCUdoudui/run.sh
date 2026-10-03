#!/bin/bash
# SCUdoudui MiniCPM5-1B official launch script
set -euo pipefail
# Launch contract for tools/ci_pipline/run_ci.py

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
MODEL_PATH="${MINICPM5_MODEL:-${MODEL_ROOT}/OpenBMB/MiniCPM5-1B}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MINICPM_OP_PROFILE="${MINICPM_OP_PROFILE:-official}"
export PYTHONPATH="${SCRIPT_DIR}/runtime/overlay:${SCRIPT_DIR}/runtime:${PYTHONPATH:-}"
# The current vLLM-SDAA compiler path cannot fake-propagate the tecoops
# pybind pointer ABI. Bind the reviewed operators eagerly at process start.
export TORCH_COMPILE_DISABLE=1

exec vllm serve "${MODEL_PATH}" \
    --served-model-name MiniCPM5-1B \
    --tensor-parallel-size 1 \
    --port "${MINICPM_PORT:-8000}" \
    --host "${MINICPM_HOST:-0.0.0.0}" \
    --dtype float16 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --max-model-len "${MINICPM_MAX_MODEL_LEN:-32768}"
