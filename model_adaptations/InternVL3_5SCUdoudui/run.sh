#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
MODEL_PATH="${INTERNVL_MODEL:-${MODEL_ROOT}/OpenGVLab/InternVL3_5-8B}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${SCRIPT_DIR}/runtime/overlay:${SCRIPT_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name InternVL3_5-8B \
    --dtype float16 \
    --tensor-parallel-size "${INTERNVL_TP_SIZE:-2}" \
    --gpu-memory-utilization "${INTERNVL_UTIL:-0.85}" \
    --max-model-len "${INTERNVL_MAX_MODEL_LEN:-4096}" \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --host "${INTERNVL_HOST:-0.0.0.0}" \
    --port "${INTERNVL_PORT:-8002}"
