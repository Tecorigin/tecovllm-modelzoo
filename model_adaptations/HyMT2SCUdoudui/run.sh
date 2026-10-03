#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
MODEL_PATH="${HY_MT2_MODEL:-${MODEL_ROOT}/Tencent-Hunyuan/Hy-MT2-1.8B}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${SCRIPT_DIR}/runtime/overlay:${SCRIPT_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name Hy-MT2-1.8B \
    --tensor-parallel-size "${HY_MT2_TP_SIZE:-1}" \
    --gpu-memory-utilization "${HY_MT2_UTIL:-0.90}" \
    --max-model-len "${HY_MT2_MAX_MODEL_LEN:-8192}" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --dtype float16 \
    --host "${HY_MT2_HOST:-0.0.0.0}" \
    --port "${HY_MT2_PORT:-8001}"
